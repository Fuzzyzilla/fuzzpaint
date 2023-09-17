//! # Actions API
//!
//! (certainly inspired by Godot's `Input`!)
//!
//! Provides an `ActionStream` to be written into by many asynchronous hotkey sources.
//! Many `ActionListener`s can then be attatched, which maintain their own state. These
//! listeners provide `ActionFrame`s that describe, on a per-listener basis, which actions
//! have been pressed, held, repeated, ect. since the last time it was listened to.

pub mod hotkeys;

#[derive(
    serde::Serialize, serde::Deserialize, Hash, PartialEq, Eq, strum::AsRefStr, Clone, Copy,
)]
pub enum Action {
    Undo,
    Redo,

    ViewportPan,
    ViewportScrub,

    LayerUp,
    LayerDown,
    LayerNew,
    LayerDelete,
}
#[derive(Clone, Copy, PartialEq, Eq)]
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

#[derive(Clone)]
struct ActionStates {
    /// Which action keys are pressed?
    held: std::collections::HashSet<Action>,
    /// Which action keys are shadowed by a more specific action key?
    shadowed: std::collections::HashSet<Action>,
    /// Which action keys have ever been shadowed in their lifetimes?
    was_ever_shadowed: std::collections::HashSet<Action>,
}
pub struct ActionStream {
    current_state: ActionStates,
    send: tokio::sync::broadcast::Sender<(ActionEvent, Action)>,
}
impl ActionStream {
    pub fn listen(&self) -> ActionListener {
        ActionListener {
            poisoned: false,
            current_state: self.current_state.clone(),
            recv: self.send.subscribe(),
        }
    }
}
pub enum ListenError {
    Poisoned,
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
        if self.poisoned {
            Err(ListenError::Poisoned)
        } else {
            let mut actions = Vec::with_capacity(self.recv.len());

            // Recieve as many actions as are available, or poison
            // and fail if lagged.
            loop {
                let recv = self.recv.try_recv();

                use tokio::sync::broadcast::error::TryRecvError;
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
    /// Return the action state immediately, without waiting for new actions.
    /// None if poisoned (though, this function will never cause it to become poisoned.)
    /// Can only be used to check if an action is currently held, not if an action has been
    /// pressed or released. use `ActionListener::frame` for that!
    pub fn now(&self) -> Option<()> {
        if self.poisoned {
            None
        } else {
            todo!()
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
    pub fn action_trigger_count(&self, action: Action) -> usize {
        let mut count = 0;

        // Keep track of if the event was shadowed as we step through.
        // Shadowed at the beginning doesn't mean we can immediately disregard it -
        // as it could have been released and then pressed again during later actions.
        let mut is_ever_shadowed = self.base_state.was_ever_shadowed.contains(&action);

        for event in self.actions.iter() {
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
                    ActionEvent::Repeat => count += 1,
                    ActionEvent::Release => count += 1,
                    // Begin shadowing
                    ActionEvent::Shadowed => is_ever_shadowed = true,
                    _ => (),
                }
            }
        }

        count
    }
    /// For actions that rely on the key being held - returns true
    /// if this action is still held by the end of the frame.
    /// Shadowed actions return false, but will resume returning true
    /// once they are unshadowed if they are still held.
    pub fn is_action_held(&self, action: Action) -> bool {
        let mut is_shadowed = self.base_state.shadowed.contains(&action);
        let mut is_held = self.base_state.held.contains(&action);

        for event in self.actions.iter() {
            // skip unrelated events
            if event.1 != action {
                continue;
            }

            match event.0 {
                ActionEvent::Press => is_held = true,
                ActionEvent::Release => is_held = false,
                ActionEvent::Repeat => (),
                ActionEvent::Shadowed => is_shadowed = true,
                ActionEvent::Unshadowed => is_shadowed = false,
            }
        }

        is_held && !is_shadowed
    }
    /// Accumulate actions into a new state representing the end of this frame.
    fn fast_forward(&self) -> ActionStates {
        let mut future = self.base_state.clone();
        for action in self.actions.iter() {
            match action {
                (ActionEvent::Press, action) => {
                    future.held.insert(*action);
                }
                (ActionEvent::Release, action) => {
                    // Upon release, shadow and ever_shadowed state get reset.
                    future.held.remove(action);
                    future.shadowed.remove(action);
                    future.was_ever_shadowed.remove(action);
                }
                (ActionEvent::Repeat, action) => {
                    // Shouldn't be necessary, but just in case!
                    future.held.insert(*action);
                }
                (ActionEvent::Shadowed, action) => {
                    future.shadowed.insert(*action);
                    future.was_ever_shadowed.insert(*action);
                }
                (ActionEvent::Unshadowed, action) => {
                    future.shadowed.remove(action);
                    // Leave `was_ever_shadowed`
                }
            }
        }

        future
    }
}
