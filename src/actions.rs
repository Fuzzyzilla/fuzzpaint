//! # Actions API
//!
//! (certainly inspired by Godot's `Input`!)
//!
//! Provides an `ActionStream` to be written into by many asynchronous hotkey sources.
//! Many `ActionListener`s can then be attatched, which maintain their own state. These
//! listeners provide `ActionFrame`s that describe, on a per-listener basis, which actions
//! have been pressed, held, repeated, ect. since the last time it was listened to.

use std::sync::Arc;

pub mod hotkeys;
pub mod winit_action_collector;

#[derive(
    serde::Serialize, serde::Deserialize, Hash, PartialEq, Eq, strum::AsRefStr, Clone, Copy, Debug,
)]
pub enum Action {
    Undo,
    Redo,

    ViewportPan,
    ViewportScrub,
    ViewportRotate,
    ViewportFlipHorizontal,

    ZoomIn,
    ZoomOut,

    Gizmo,
    Brush,
    Erase,
    Lasso,

    BrushSizeUp,
    BrushSizeDown,

    LayerUp,
    LayerDown,
    LayerNew,
    LayerDelete,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActionEvent {
    Press,
    /// The action was held long enough that the key is being strobed by the OS.
    Repeat,
    Release,

    /// The action was pressed, but another action with the same keys but stricter modifiers overwrote it.
    /// For example, the key sequence Ctrl + S + Shift could result in:
    ///
    /// * Press Save
    ///   * Shadow Save
    ///     * Press Save as
    ///     * Release Save as
    ///   * Unshadow Save
    /// * Release Save
    ///
    /// It is up to the listener to figure out what the user meant x3
    /// Typically a (non-holding) action that has ever been shadowed should be ignored.
    Shadowed,
    Unshadowed,
}

#[derive(Clone, Default)]
struct ActionStates {
    /// Which action keys are pressed?
    held: std::collections::HashSet<Action>,
    /// Which action keys are shadowed by a more specific action key?
    shadowed: std::collections::HashSet<Action>,
    /// Which action keys have ever been shadowed in their lifetimes?
    was_ever_shadowed: std::collections::HashSet<Action>,
}
impl ActionStates {
    fn push(&mut self, event: ActionEvent, action: Action) {
        match event {
            ActionEvent::Press | ActionEvent::Repeat => {
                self.held.insert(action);
            }
            ActionEvent::Release => {
                // Upon release, shadow and ever_shadowed state get reset.
                self.held.remove(&action);
                self.shadowed.remove(&action);
                self.was_ever_shadowed.remove(&action);
            }
            ActionEvent::Shadowed => {
                self.shadowed.insert(action);
                self.was_ever_shadowed.insert(action);
            }
            ActionEvent::Unshadowed => {
                self.shadowed.remove(&action);
                // Leave `was_ever_shadowed`
            }
        }
    }
}

/// Create a send/recieve pair. The recieve side can have as many listeners as it wants spawned via
/// `ActionStream::listen`, but sending is a unique duty.
#[must_use]
pub fn create_action_stream() -> (ActionSender, ActionStream) {
    let (send, recv) = tokio::sync::broadcast::channel(32);

    let current_state = (ActionStates::default(), recv);
    let current_state: parking_lot::RwLock<_> = current_state.into();
    let current_state = Arc::new(current_state);

    (
        ActionSender {
            send,
            current_state: Arc::downgrade(&current_state),
        },
        ActionStream { current_state },
    )
}

/// Holds the internal state of an action channel
type SenderStateLock = parking_lot::RwLock<(
    ActionStates,
    tokio::sync::broadcast::Receiver<(ActionEvent, Action)>,
)>;

pub struct ActionSender {
    // Weak, as we can stop carrying/updating this info when it stops being possible to create listeners
    // (ie, when the holder of the Arc is dropped)
    current_state: std::sync::Weak<SenderStateLock>,
    send: tokio::sync::broadcast::Sender<(ActionEvent, Action)>,
}
impl ActionSender {
    pub fn press(&self, action: Action) {
        self.push(ActionEvent::Press, action);
    }
    pub fn release(&self, action: Action) {
        self.push(ActionEvent::Release, action);
    }
    pub fn repeat(&self, action: Action) {
        self.push(ActionEvent::Repeat, action);
    }
    pub fn shadow(&self, action: Action) {
        self.push(ActionEvent::Shadowed, action);
    }
    pub fn unshadow(&self, action: Action) {
        self.push(ActionEvent::Unshadowed, action);
    }
    fn oneshot(&self, action: Action) {
        // Double locks, could speed up.
        self.press(action);
        self.release(action);
    }
    fn push(&self, event: ActionEvent, action: Action) {
        match self.current_state.upgrade() {
            Some(rw) => {
                let mut write = rw.write();
                write.0.push(event, action);
                // Returns err if no listeners.
                let _ = self.send.send((event, action));
                // The send must occur while the lock is held.
                // The compiler doens't know this, but I hope that
                // this will enforce it: (todo: verify)
                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
                // Pretend the lock is still used, and drop it.
                drop(std::hint::black_box(write));
            }
            None => {
                // Returns err if no listeners.
                let _ = self.send.send((event, action));
            }
        }
    }
}

pub struct ActionStream {
    current_state: Arc<SenderStateLock>,
}
impl ActionStream {
    #[must_use]
    pub fn listen(&self) -> ActionListener {
        // All is locked behind RwLock to prevent subtle race condition:
        // recv and current state are deeply intertwined, and
        // it's important that recv's "start" aligns with the exact moment
        // `current_state` is captured from. Thus, we lock the current state,
        // and make the reciever while it's locked to ensure it hasn't been changed.

        let lock = self.current_state.read();
        let current_state = lock.0.clone();
        let recv = lock.1.resubscribe();

        ActionListener {
            poisoned: false,
            current_state,
            recv,
        }
    }
}
#[derive(thiserror::Error, Debug)]
pub enum ListenError {
    #[error("Listener poisoned")]
    Poisoned,
    #[error("Listener closed")]
    Closed,
}
pub struct ActionListener {
    /// Listener becomes poisoned if it lags to the point that actions are missed,
    /// as the action state will become desync'd.
    poisoned: bool,
    current_state: ActionStates,
    recv: tokio::sync::broadcast::Receiver<(ActionEvent, Action)>,
}
impl ActionListener {
    /// Get the actions performed since the last call to this listener's
    /// `frame`. Returns None if it has been too long since the last poll and data was lost.
    /// In the case where None is returned, this listener is poisoned and will always return None -
    /// it must be re-acquired from the main `ActionStream` to repair.
    pub fn frame(&mut self) -> Result<ActionFrame, ListenError> {
        use tokio::sync::broadcast::error::TryRecvError;

        if self.poisoned {
            Err(ListenError::Poisoned)
        } else {
            let mut actions = Vec::with_capacity(self.recv.len());

            // Recieve as many actions as are available, or poison
            // and fail if lagged.
            loop {
                let recv = self.recv.try_recv();

                match recv {
                    Ok(action) => actions.push(action),
                    Err(TryRecvError::Closed | TryRecvError::Empty) => break,
                    Err(TryRecvError::Lagged(..)) => {
                        self.poisoned = true;
                        return Err(ListenError::Poisoned);
                    }
                }
            }

            let action_frame = ActionFrame {
                base_state: self.current_state.clone(),
                actions,
            };

            // Accumulate the actions into the base state for next frame.
            self.current_state = action_frame.fast_forward();

            Ok(action_frame)
        }
    }
    /// Same as frame, but will wait for data to arrive instead of returning an empty
    /// frame. Cancel safe.
    pub async fn wait_frame(&mut self) -> Result<ActionFrame, ListenError> {
        use tokio::sync::broadcast::error::RecvError;
        use tokio::sync::broadcast::error::TryRecvError;

        if self.poisoned {
            Err(ListenError::Poisoned)
        } else {
            let mut actions = Vec::with_capacity(self.recv.len());

            // Recieve as many actions as are available, or poison
            // and fail if lagged.

            // First, asynchronously read, stalling until data available.
            let recv = self.recv.recv().await;
            match recv {
                Ok(action) => actions.push(action),
                Err(RecvError::Closed) => return Err(ListenError::Closed),
                Err(RecvError::Lagged(..)) => {
                    self.poisoned = true;
                    return Err(ListenError::Poisoned);
                }
            }

            // Then, try to recieve as many more as available without waiting.
            loop {
                let recv = self.recv.try_recv();

                match recv {
                    Ok(action) => actions.push(action),
                    Err(TryRecvError::Closed | TryRecvError::Empty) => break,
                    Err(TryRecvError::Lagged(..)) => {
                        self.poisoned = true;
                        return Err(ListenError::Poisoned);
                    }
                }
            }

            let action_frame = ActionFrame {
                base_state: self.current_state.clone(),
                actions,
            };

            // Accumulate the actions into the base state for next frame.
            self.current_state = action_frame.fast_forward();

            Ok(action_frame)
        }
    }
}
pub struct ActionFrame {
    base_state: ActionStates,
    actions: Vec<(ActionEvent, Action)>,
}
impl ActionFrame {
    /// Count the number of times this action was triggered since the last frame.
    /// For actions that trigger once on press - like undo.
    /// Will return multiple counts if the OS repeats the key, +1 on the releasing edge.
    ///
    /// If the action has ever became shadowed in it's lifetime, it will stop
    /// counting triggers permanently until released and repressed.
    #[must_use]
    pub fn action_trigger_count(&self, action: Action) -> usize {
        let mut count = 0;

        // Keep track of if the event was shadowed as we step through.
        // Shadowed at the beginning doesn't mean we can immediately disregard it -
        // as it could have been released and then pressed again during later actions.
        let mut is_ever_shadowed = self.base_state.was_ever_shadowed.contains(&action);

        for event in &self.actions {
            // Skip unrelated events
            if event.1 != action {
                continue;
            };

            // Tiny state machine - if shadowed, we only care about release events.
            if is_ever_shadowed {
                // Only a release can make us start counting triggers again.
                // Release event during shadow does not count towards triggers
                if event.0 == ActionEvent::Release {
                    is_ever_shadowed = false;
                }
            } else {
                // Not shadowed - count up release and repeat events.
                match event.0 {
                    ActionEvent::Repeat | ActionEvent::Release => count += 1,
                    // Begin shadowing
                    ActionEvent::Shadowed => is_ever_shadowed = true,
                    _ => (),
                }
            }
        }

        count
    }
    /// Query whether anything happened this frame. `ActionListener::frame` will return
    /// a frame with no events if queried before anything new comes in.
    #[must_use]
    pub fn changed(&self) -> bool {
        !self.actions.is_empty()
    }
    /// For actions that rely on the key being held - returns true
    /// if this action is still held by the end of the frame.
    /// Shadowed actions return false, but will resume returning true
    /// once they are unshadowed if they are still held.
    #[must_use]
    pub fn is_action_held(&self, action: Action) -> bool {
        let mut is_shadowed = self.base_state.shadowed.contains(&action);
        let mut is_held = self.base_state.held.contains(&action);

        for event in &self.actions {
            // skip unrelated events
            if event.1 != action {
                continue;
            }

            match event.0 {
                ActionEvent::Press | ActionEvent::Repeat => is_held = true,
                ActionEvent::Release => is_held = false,
                ActionEvent::Shadowed => is_shadowed = true,
                ActionEvent::Unshadowed => is_shadowed = false,
            }
        }

        is_held && !is_shadowed
    }
    /// Accumulate actions into a new state representing the end of this frame.
    fn fast_forward(&self) -> ActionStates {
        let mut future = self.base_state.clone();
        for (event, action) in &self.actions {
            future.push(*event, *action);
        }

        future
    }
}
