use fuzzpaint_types::stroke::{Archetype, aos::StrokeSlice};

#[non_exhaustive]
#[derive(Copy, Clone, Debug)]
pub enum StylusAxis {
    PosX,
    PosY,
    TiltX,
    TiltY,
    Pressure,
    Dist,
}

// Multi-seat-aware winit and octotablet event funnel of doom.
//
// Collates all of the pointer devices into a single virtual pointer device for
// Egui, as well as collecting the drawing motions.
//
// * All platforms need multi-seat awareness, because octotablet! Multiple
//   styluses can be used at once, even on inherently single-seat systems
//   (windows).
// * If there are any tools or touches are "In":
//      * Ignore all mice entirely until all are out.
//      * Systems cannot be trusted with mutliple cursors on one seat (windows,
//   even some linux DEs) and try to emulate tools from mice and mice from tools
//   and its a nightmare.
//      * Emulate mouse (for EGUI) from the first IN tool.
// * Otherwise:
//      * emulate tools from mice.
//      * EGUI is single cursor, which cursor to forward???

// Down tools > Down touches > up tools > mice.

// We only ever let egui know about one mouse and touch device, named by these:
// All other devices get erased into these ids.
const EMULATED_DEVICE_ID: winit::event::DeviceId = winit::event::DeviceId::dummy();
const EMULATED_TOUCH_IDX: u64 = 0;

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub enum ToolID {
    HardwareSerial(octotablet::tool::HardwareID),
    Opaque(octotablet::tool::ID),
}
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct TouchID(winit::event::DeviceId, u64);
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MouseID(winit::event::DeviceId);

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub enum DeviceID {
    // In order of quality, highest to lowest:
    Tool(ToolID),
    Mouse(MouseID),
    Touch(TouchID),
}
impl From<ToolID> for DeviceID {
    fn from(value: ToolID) -> Self {
        Self::Tool(value)
    }
}
impl From<MouseID> for DeviceID {
    fn from(value: MouseID) -> Self {
        Self::Mouse(value)
    }
}
impl From<TouchID> for DeviceID {
    fn from(value: TouchID) -> Self {
        Self::Touch(value)
    }
}

#[derive(Hash, PartialEq, Eq, Debug)]
pub struct SubuserID(DeviceID);

fn or_pose(prev: octotablet::axis::Pose, next: octotablet::axis::Pose) -> octotablet::axis::Pose {
    octotablet::axis::Pose {
        position: next.position,
        distance: next
            .distance
            .get()
            .or(prev.distance.get())
            .try_into()
            .unwrap(),
        pressure: next
            .pressure
            .get()
            .or(prev.pressure.get())
            .try_into()
            .unwrap(),
        button_pressure: next
            .button_pressure
            .get()
            .or(prev.button_pressure.get())
            .try_into()
            .unwrap(),
        tilt: next.tilt.or(prev.tilt),
        roll: next.roll.get().or(prev.roll.get()).try_into().unwrap(),
        wheel: next.wheel.or(prev.wheel),
        slider: next.slider.get().or(prev.slider.get()).try_into().unwrap(),
        contact_size: next.contact_size.or(prev.contact_size),
    }
}

pub struct ToolState {
    // Always in, otherwise it wouldn't be here!
    over: octotablet::tablet::ID,
    is_eraser: bool,
    down: bool,
    right_click: bool,
    pose: octotablet::axis::Pose,
}
impl ToolState {
    fn down_buttons(&self) -> impl Iterator<Item = winit::event::MouseButton> {
        [
            self.down.then_some(winit::event::MouseButton::Left),
            self.right_click.then_some(winit::event::MouseButton::Right),
        ]
        .into_iter()
        .flatten()
    }
    fn transition_away(&self, events: &mut Vec<winit::event::WindowEvent>) {
        // Release all buttons.
        for button in self.down_buttons() {
            events.push(winit::event::WindowEvent::MouseInput {
                device_id: EMULATED_DEVICE_ID,
                state: winit::event::ElementState::Released,
                button,
            });
        }

        // Leave :3
        events.push(winit::event::WindowEvent::CursorLeft {
            device_id: EMULATED_DEVICE_ID,
        });
    }
    fn transition_into(&self, events: &mut Vec<winit::event::WindowEvent>) {
        // Enter
        events.push(winit::event::WindowEvent::CursorEntered {
            device_id: EMULATED_DEVICE_ID,
        });

        // Teleport to location
        events.push(winit::event::WindowEvent::CursorMoved {
            device_id: EMULATED_DEVICE_ID,
            position: self.pose.position.into(),
        });
        // Press all the buttons
        for button in self.down_buttons() {
            events.push(winit::event::WindowEvent::MouseInput {
                device_id: EMULATED_DEVICE_ID,
                state: winit::event::ElementState::Released,
                button,
            });
        }
        // And report the force of the press.
        if let Some(pressure) = self.pose.pressure.get() {
            events.push(winit::event::WindowEvent::TouchpadPressure {
                device_id: EMULATED_DEVICE_ID,
                pressure,
                // Dont have this.
                stage: 0,
            });
        }
    }
}
#[derive(Default)]
pub struct TouchState {
    // Always down and in, otherwise it wouldn't be here! Touches have no
    // concept of "in but up".
    position: [f32; 2],
    pressure: Option<f32>,
    // Altitude from horizontal, radians, [0, PI/2.0].
    altitude: Option<f32>,
}
impl TouchState {
    fn winit_force(&self) -> Option<winit::event::Force> {
        match (self.pressure, self.altitude) {
            (None, None) => None,
            (Some(_), _) | (_, Some(_)) => Some(winit::event::Force::Calibrated {
                force: self.pressure.unwrap_or(1.0) as _,
                max_possible_force: 1.0,
                altitude_angle: self.altitude.map(|x| x as _),
            }),
        }
    }
    fn transition_away(&self, events: &mut Vec<winit::event::WindowEvent>) {
        // Cancel the touch.
        events.push(winit::event::WindowEvent::Touch(winit::event::Touch {
            device_id: EMULATED_DEVICE_ID,
            // Transitions away only if something pre-empted it. Thus, we should
            // cancel, not End.
            phase: winit::event::TouchPhase::Cancelled,
            location: self.position.into(),
            force: self.winit_force(),
            id: EMULATED_TOUCH_IDX,
        }));
    }
    fn transition_into(&self, events: &mut Vec<winit::event::WindowEvent>) {
        events.push(winit::event::WindowEvent::Touch(winit::event::Touch {
            device_id: EMULATED_DEVICE_ID,
            phase: winit::event::TouchPhase::Started,
            location: self.position.into(),
            force: self.winit_force(),
            id: EMULATED_TOUCH_IDX,
        }));
    }
}
#[derive(Default)]
pub struct MouseState {
    // Always in, otherwise it wouldn't be here!
    down_buttons: hashbrown::HashSet<winit::event::MouseButton>,
    // No capabilities other than position.
    position: [f32; 2],
    pressure: Option<f32>,
}
impl MouseState {
    fn logically_down(&self) -> bool {
        self.down_buttons.contains(&winit::event::MouseButton::Left)
    }
    fn transition_away(&self, events: &mut Vec<winit::event::WindowEvent>) {
        // Release all buttons.
        for button in &self.down_buttons {
            events.push(winit::event::WindowEvent::MouseInput {
                device_id: EMULATED_DEVICE_ID,
                state: winit::event::ElementState::Released,
                button: *button,
            });
        }

        // Leave :3
        events.push(winit::event::WindowEvent::CursorLeft {
            device_id: EMULATED_DEVICE_ID,
        });
    }
    fn transition_into(&self, events: &mut Vec<winit::event::WindowEvent>) {
        // Enter
        events.push(winit::event::WindowEvent::CursorEntered {
            device_id: EMULATED_DEVICE_ID,
        });

        // Teleport to location
        events.push(winit::event::WindowEvent::CursorMoved {
            device_id: EMULATED_DEVICE_ID,
            position: self.position.into(),
        });
        // Press all the buttons
        for button in &self.down_buttons {
            events.push(winit::event::WindowEvent::MouseInput {
                device_id: EMULATED_DEVICE_ID,
                state: winit::event::ElementState::Released,
                button: *button,
            });
        }
        // And report the force of the press.
        if let Some(pressure) = self.pressure {
            events.push(winit::event::WindowEvent::TouchpadPressure {
                device_id: EMULATED_DEVICE_ID,
                pressure,
                // Lost info. oh well.
                stage: 0,
            });
        }
    }
}

pub struct Stroke {
    generation: u8,
}

#[derive(Default)]
pub struct PointerBridge {
    // Keep track of the full state of all pointing devices at all time. Thus,
    // if they become main, we know what they're up to, and we can emulate the
    // events needed to transition from one pointer's state to another.
    in_tools: hashbrown::HashMap<ToolID, ToolState>,
    active_touches: hashbrown::HashMap<TouchID, TouchState>,
    in_mice: hashbrown::HashMap<MouseID, MouseState>,

    ongoing_strokes: hashbrown::HashMap<DeviceID, ()>,
    new_strokes: Vec<StrokeListener>,

    synthetic_events: Vec<winit::event::WindowEvent>,
}
impl PointerBridge {
    fn use_mouse_for_egui(&self) -> bool {
        self.in_tools.is_empty() && self.active_touches.is_empty()
    }
    fn use_touch_for_egui(&self) -> bool {
        self.in_tools.is_empty()
    }
    fn get_egui_main(&self) -> Option<DeviceID> {
        // We don't care which (hence the unspecified iter order) so long as the
        // tools > touches > mice priority is respected.
        self.in_tools
            .keys()
            .next()
            .cloned()
            .map(Into::into)
            .or_else(|| self.active_touches.keys().next().cloned().map(Into::into))
            .or_else(|| self.in_mice.keys().next().cloned().map(Into::into))
    }
    // Get or create the state for the given device. If this device just became
    // the main, a transition from the previous main is automatically performed.
    fn mouse_or_default(&mut self, id: MouseID) -> &mut MouseState {
        let prev_main = self.get_egui_main();
        let _ = self.in_mice.entry(id.clone()).or_default();
        if self.get_egui_main() == Some(DeviceID::Mouse(id.clone())) {
            // This just became the main, so transition away from the old main.
            if let Some(prev_main) = prev_main {
                self.transition_away(prev_main);
            }
            // FIXME: improperly reports a move to 0,0
            self.transition_into(DeviceID::Mouse(id.clone()));
        }
        // :< unfortunate double-get.
        self.in_mice.get_mut(&id).unwrap()
    }
    // Get or create the state for the given device. If this device just became
    // the main, a transition from the previous main is automatically performed.
    fn touch_or_default(&mut self, id: TouchID) -> &mut TouchState {
        let prev_main = self.get_egui_main();
        let _ = self.active_touches.entry(id.clone()).or_default();
        if self.get_egui_main() == Some(DeviceID::Touch(id.clone())) {
            // This just became the main, so transition away from the old main.
            if let Some(prev_main) = prev_main {
                self.transition_away(prev_main);
            }
            // FIXME: improperly reports a move to 0,0
            self.transition_into(DeviceID::Touch(id.clone()));
        }
        // :< unfortunate double-get.
        self.active_touches.get_mut(&id).unwrap()
    }
    // Get or create the state for the given device. If this device just became
    // the main, a transition from the previous main is automatically performed.
    // This differs from the other types as the ToolState is !Default.
    fn insert_tool(&mut self, id: ToolID, state: ToolState) -> &mut ToolState {
        let prev_main = self.get_egui_main();
        let _ = self.in_tools.insert(id.clone(), state);
        if self.get_egui_main() == Some(DeviceID::Tool(id.clone())) {
            // This just became the main, so transition away from the old main.
            if let Some(prev_main) = prev_main {
                self.transition_away(prev_main);
            }
            // FIXME: improperly reports a move to 0,0
            self.transition_into(DeviceID::Tool(id.clone()));
        }
        // :< unfortunate double-get.
        self.in_tools.get_mut(&id).unwrap()
    }
    fn remove(&mut self, id: impl Into<DeviceID>) {
        let id = id.into();
        let was_main = self.get_egui_main() == Some(id.clone());
        // Was main, transition away from it.
        if was_main {
            self.transition_away(id.clone());
        }
        match id {
            DeviceID::Mouse(m) => {
                self.in_mice.remove(&m);
            }
            DeviceID::Tool(t) => {
                self.in_tools.remove(&t);
            }
            DeviceID::Touch(t) => {
                self.active_touches.remove(&t);
            }
        }
        // Transition into a new main.
        if was_main && let Some(new_main) = self.get_egui_main() {
            self.transition_into(new_main);
        }
    }
    /// Produce synthetic events of the given id leaving.
    fn transition_away(&mut self, id: DeviceID) {
        match id {
            DeviceID::Mouse(m) => {
                if let Some(mouse) = self.in_mice.get(&m) {
                    mouse.transition_away(&mut self.synthetic_events);
                }
            }
            DeviceID::Tool(t) => {
                if let Some(tool) = self.in_tools.get(&t) {
                    tool.transition_away(&mut self.synthetic_events);
                }
            }
            DeviceID::Touch(t) => {
                if let Some(touch) = self.active_touches.get(&t) {
                    touch.transition_away(&mut self.synthetic_events);
                }
            }
        }
    }
    /// Produce synthetic events of the given device entering and becoming its
    /// current state.
    fn transition_into(&mut self, id: DeviceID) {
        match id {
            DeviceID::Mouse(m) => {
                if let Some(mouse) = self.in_mice.get(&m) {
                    mouse.transition_into(&mut self.synthetic_events);
                }
            }
            DeviceID::Tool(t) => {
                if let Some(tool) = self.in_tools.get(&t) {
                    tool.transition_into(&mut self.synthetic_events);
                }
            }
            DeviceID::Touch(t) => {
                if let Some(touch) = self.active_touches.get(&t) {
                    touch.transition_into(&mut self.synthetic_events);
                }
            }
        }
    }
    fn is_egui_main(&self, id: impl Into<DeviceID>) -> bool {
        match id.into() {
            // No device overshadows it, and it's the "first" mouse by some
            // arbitrary metric of first-ness.
            DeviceID::Mouse(m) => {
                self.use_mouse_for_egui() && self.in_mice.keys().next() == Some(&m)
            }
            // Is "first" tool by some arbitrary metric of first-ness.
            DeviceID::Tool(t) => self.in_tools.keys().next() == Some(&t),
            // Is "first" touch by some arbitrary metric of first-ness.
            DeviceID::Touch(t) => {
                self.use_touch_for_egui() && self.active_touches.keys().next() == Some(&t)
            }
        }
    }
    // Push a winit event, returning zero or more synthetic winit events to
    // forward to egui_winit.
    pub fn push_winit(
        &mut self,
        event: winit::event::WindowEvent,
    ) -> impl Iterator<Item = winit::event::WindowEvent> {
        use winit::event::{ElementState, WindowEvent as Event};

        match event {
            Event::MouseInput {
                device_id,
                state,
                button,
            } => {
                let mouse = self.mouse_or_default(MouseID(device_id));
                match state {
                    ElementState::Pressed => mouse.down_buttons.insert(button),
                    ElementState::Released => mouse.down_buttons.remove(&button),
                };
                if self.is_egui_main(MouseID(device_id)) {
                    self.synthetic_events.push(Event::MouseInput {
                        device_id: EMULATED_DEVICE_ID,
                        state,
                        button,
                    });
                }
            }
            Event::CursorEntered { device_id } => {
                // Automatically creates a synthetic Enter if need be.
                self.mouse_or_default(MouseID(device_id));
            }
            Event::CursorLeft { device_id } => {
                // Automatically creates a synthetic Leave if need be.
                self.remove(MouseID(device_id));
            }
            Event::CursorMoved {
                device_id,
                position,
            } => {
                self.in_mice.entry(MouseID(device_id)).or_default().position =
                    position.cast::<f32>().into();
                if self.is_egui_main(MouseID(device_id)) {
                    self.synthetic_events.push(Event::CursorMoved {
                        device_id: EMULATED_DEVICE_ID,
                        position,
                    });
                }
            }
            Event::Touch(t) => {
                use winit::event::TouchPhase;
                // Fixme: Manually detect gestures.
                let id = TouchID(t.device_id, t.id);
                match t.phase {
                    TouchPhase::Cancelled => {
                        // Automatically creates a synthetic Cancel if need be.
                        self.remove(id);
                    }
                    TouchPhase::Ended => {
                        // Manually do it, since the `remove` transition uses a
                        // Cancel.
                        let was_main = self.is_egui_main(id.clone());
                        if was_main {
                            // End the touch normally
                            self.synthetic_events
                                .push(Event::Touch(winit::event::Touch {
                                    device_id: EMULATED_DEVICE_ID,
                                    id: EMULATED_TOUCH_IDX,
                                    ..t
                                }));
                        }
                        self.active_touches.remove(&id);
                        // If this is the main that is ending, transition to the new main.
                        if was_main && let Some(new_main) = self.get_egui_main() {
                            self.transition_into(new_main);
                        }
                    }
                    _ => {
                        let touch = self.touch_or_default(id.clone());
                        touch.position = t.location.cast::<f32>().into();
                        if let Some(force) = t.force {
                            match force {
                                // NOT Equivalent to f.normalized(), that is perpendicular force whereas we want axial force.
                                winit::event::Force::Calibrated {
                                    force,
                                    max_possible_force,
                                    altitude_angle,
                                } => {
                                    touch.pressure = Some((force / max_possible_force) as f32);
                                    touch.altitude =
                                        altitude_angle.map(|angle| angle as f32).or(touch.altitude);
                                }
                                winit::event::Force::Normalized(n) => {
                                    touch.pressure = Some(n as f32)
                                }
                            }
                        }
                        if self.is_egui_main(id) {
                            self.synthetic_events
                                .push(Event::Touch(winit::event::Touch {
                                    device_id: EMULATED_DEVICE_ID,
                                    id: EMULATED_TOUCH_IDX,
                                    ..t
                                }));
                        }
                    }
                }
            }
            Event::TouchpadPressure { .. } => {
                // Fixme: is this a mouse or a touch device?
                // This is mac only, and we don't support mac anyway UWU
                todo!();
                /*
                let mouse = self.mouse_or_default(MouseID(device_id));
                mouse.pressure = Some(pressure);
                if self.is_egui_main(MouseID(device_id)) {
                    self.synthetic_events.push(Event::TouchpadPressure {
                        device_id: EMULATED_DEVICE_ID,
                        pressure,
                        stage,
                    });
                }*/
            }
            // Uhh i dont think this is relevant?
            // Event::MouseWheel { device_id, delta, phase } => todo!();
            // Irrelevant to us, forward as-is.
            e => self.synthetic_events.push(e),
        }
        self.synthetic_events.drain(..)
    }
    // Push an octotablet event, returning zero or more synthetic winit events
    // to forward to egui_winit.
    pub fn push_octotablet(
        &mut self,
        event: octotablet::events::Event<'_>,
    ) -> impl Iterator<Item = winit::event::WindowEvent> {
        use octotablet::events::{Event, TabletEvent, ToolEvent};
        match event {
            Event::Tablet { tablet, event } => match event {
                TabletEvent::Removed => {
                    let tablet_id = tablet.id();

                    // Find all tools over this tablet and remove em!
                    let to_remove = self
                        .in_tools
                        .iter()
                        .filter_map(|(tool_id, state)| {
                            (state.over == tablet_id).then_some(tool_id.clone())
                        })
                        .collect::<Vec<_>>();
                    // if We're removing the current main. Transition away from
                    // it first.
                    let current_main = self.get_egui_main();
                    let did_transition = if let Some(DeviceID::Tool(current_main)) = current_main
                        && to_remove.contains(&current_main)
                    {
                        self.transition_away(DeviceID::Tool(current_main));
                        true
                    } else {
                        false
                    };
                    // remove!
                    for id in to_remove {
                        self.in_tools.remove(&id);
                    }
                    // If we removed the current main, then transition to
                    // whatever the new main is.
                    if did_transition && let Some(new_main) = self.get_egui_main() {
                        self.transition_into(new_main);
                    }
                }
                TabletEvent::Added => {
                    if let Some(name) = &tablet.name {
                        log::info!("Added tablet {name:?}");
                    } else {
                        log::info!("Added tablet {:?}", tablet.id());
                    }
                }
            },
            Event::Tool { tool, event } => {
                let id = if let Some(id) = tool.hardware_id {
                    ToolID::HardwareSerial(id)
                } else {
                    ToolID::Opaque(tool.id())
                };
                match event {
                    ToolEvent::In { tablet } => {
                        self.insert_tool(
                            id,
                            ToolState {
                                over: tablet.id(),
                                is_eraser: tool.tool_type == Some(octotablet::tool::Type::Eraser),
                                down: false,
                                right_click: false,
                                pose: octotablet::axis::Pose::default(),
                            },
                        );
                    }
                    ToolEvent::Out | ToolEvent::Removed => {
                        self.remove(id);
                    }
                    ToolEvent::Down => {
                        if let Some(state) = self.in_tools.get_mut(&id) {
                            state.down = true;
                            if self.is_egui_main(id) {
                                self.synthetic_events
                                    .push(winit::event::WindowEvent::MouseInput {
                                        device_id: EMULATED_DEVICE_ID,
                                        state: winit::event::ElementState::Pressed,
                                        button: winit::event::MouseButton::Left,
                                    });
                            }
                        };
                    }
                    ToolEvent::Up => {
                        if let Some(state) = self.in_tools.get_mut(&id) {
                            state.down = false;
                            if self.is_egui_main(id) {
                                self.synthetic_events
                                    .push(winit::event::WindowEvent::MouseInput {
                                        device_id: EMULATED_DEVICE_ID,
                                        state: winit::event::ElementState::Released,
                                        button: winit::event::MouseButton::Left,
                                    });
                            }
                        }
                    }
                    ToolEvent::Pose(p) => {
                        if let Some(state) = self.in_tools.get_mut(&id) {
                            state.pose = or_pose(state.pose, p);
                            if self.is_egui_main(id) {
                                self.synthetic_events.push(
                                    winit::event::WindowEvent::CursorMoved {
                                        device_id: EMULATED_DEVICE_ID,
                                        position: p.position.into(),
                                    },
                                );
                            }
                        }
                    }
                    ToolEvent::Button {
                        button_id: _,
                        pressed,
                    } => {
                        // Treat all other buttons as right-click :3c
                        if let Some(state) = self.in_tools.get_mut(&id) {
                            state.right_click = pressed;
                            if self.is_egui_main(id) {
                                self.synthetic_events
                                    .push(winit::event::WindowEvent::MouseInput {
                                        device_id: EMULATED_DEVICE_ID,
                                        state: if pressed {
                                            winit::event::ElementState::Pressed
                                        } else {
                                            winit::event::ElementState::Released
                                        },
                                        button: winit::event::MouseButton::Right,
                                    });
                            }
                        }
                    }
                    _ => (),
                }
            }
            Event::Pad { .. } => (),
        }
        self.synthetic_events.drain(..)
    }
    /// Take the newly started strokes since the last time this function was
    /// called. Once the listener is taken, there is no way to access that
    /// stroke operation for the remainder of its lifetime.
    ///
    /// *New listeners should be polled in order,* as some of these listeners
    /// may represent already-complete strokes.
    pub fn take_new_listeners(&mut self) -> Vec<StrokeListener> {
        std::mem::take(&mut self.new_strokes)
    }
    /// Get the pose of the current primary pointer. You should check if egui is
    /// eating this hover first.
    ///
    /// Returns None if there is no primary pointer or if it is down (not
    /// hovering).
    pub fn primary_hover(&self) -> Option<Hover> {
        let main = self.get_egui_main()?;
        match main {
            DeviceID::Mouse(m) => {
                let mouse = self.in_mice.get(&m)?;
                (!mouse.logically_down()).then_some(Hover {
                    position: mouse.position.into(),
                    distance: None,
                })
            }
            DeviceID::Tool(t) => {
                let tool = self.in_tools.get(&t)?;
                (!tool.down).then_some(Hover::from_pose(&tool.pose))
            }
            DeviceID::Touch(_) => {
                // Touches cannot hover.
                None
            }
        }
    }
    /// Get a copy of **non-primary** pointers hovering the window. Subject to
    /// filtering (e.g. mice will never be reported while there is any stylus in
    /// range.)
    pub fn auxiliary_hovers(&self) -> Vec<Hover> {
        let mut vec = Vec::new();
        let Some(main) = self.get_egui_main() else {
            return vec;
        };

        // Ignore mice if there are active tools or touches.
        if self.in_tools.is_empty() && self.active_touches.is_empty() {
            // Collect mice.
            for (id, mouse) in &self.in_mice {
                // Only collect if not the main pointer and not currently down.
                if !mouse.logically_down() && main != DeviceID::Mouse(id.clone()) {
                    vec.push(Hover {
                        position: mouse.position.into(),
                        distance: None,
                    });
                }
            }
        } else {
            // Collect styluses. No need to collect touches, as touches have no
            // concept of a hover.
            for (id, tool) in &self.in_tools {
                // Only collect if not the main pointer and not currently down.
                if !tool.down && main != DeviceID::Tool(id.clone()) {
                    vec.push(Hover::from_pose(&tool.pose));
                }
            }
        }
        vec
    }
}

/// An auxiliary pointer hovering the window.
pub struct Hover {
    /// Logical pixels.
    position: ultraviolet::Vec2,
    /// Normalized
    distance: Option<f32>,
}
impl Hover {
    /// Get the position of the hover, in logical pixels.
    pub fn position(&self) -> ultraviolet::Vec2 {
        self.position
    }
    /// Get the perpenicular distance from the screen, normalized.
    pub fn distance(&self) -> Option<f32> {
        self.distance
    }
    fn from_pose(pose: &octotablet::axis::Pose) -> Self {
        Self {
            position: pose.position.into(),
            distance: pose.distance.get(),
        }
    }
}

fn archetype_of(p: &octotablet::axis::Pose) -> Archetype {
    [
        Some(Archetype::POSITION),
        p.tilt.is_some().then_some(Archetype::TILT),
        p.distance.get().is_some().then_some(Archetype::DISTANCE),
        p.pressure.get().is_some().then_some(Archetype::PRESSURE),
        p.roll.get().is_some().then_some(Archetype::ROLL),
        p.wheel.is_some().then_some(Archetype::WHEEL),
    ]
    .iter()
    .flatten()
    // Rust syntax looks funny sometimes :3
    .fold(Archetype::empty(), |acc, &item| acc | item)
}
/// Collect many poses into a stroke slice.
fn collect(
    poses: impl Iterator<Item = octotablet::axis::Pose>,
    mut base_archetype: Archetype,
    stage: &'_ mut Vec<u32>,
) -> StrokeSlice<'_> {
    stage.clear();
    for pose in poses {
        let archetype = archetype_of(&pose);
        if !base_archetype.contains(archetype) {
            // Expand to make room for the new archetype axes.
        }
    }
    todo!()
}

pub enum Ended {
    /// The stroke was retroactively cancelled, and the interaction should be
    /// discarded.
    Cancelled,
    /// The stroke is finished and should be committed.
    Ended,
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct StrokeListenerID {
    device: DeviceID,
    generation: u8,
}
/// Stream to listen for input from a device for the duration of a
/// click-and-drag. This listener has no capability to query down-ness of the
/// pointer, as a Stroke is by definition always logically down until it ends.
pub struct StrokeListener {
    id: StrokeListenerID,
    archetype: Archetype,
    position: ultraviolet::Vec2,
    stream: std::sync::mpsc::Receiver<octotablet::axis::Pose>,
    staging: Vec<u32>,
}
impl StrokeListener {
    /// Get an identifier for this listener. Identifiers may be reused after the
    /// previous listener of that ID is dropped. *It is important to drop a
    /// listener soon after it is no longer needed to avoid ID exhaustion.*
    pub fn id(&self) -> StrokeListenerID {
        self.id.clone()
    }
    /// Get the maximum possible archetype for the device creating this stroke.
    /// If a pointer *might* report values for the given archetype, it should be
    /// reported here even if it isn't currently.
    ///
    /// This is only a best-effort hint. The values returned by
    /// [`Self::take_stroke`] may differ.
    pub fn maximal_archetype(&self) -> Archetype {
        self.archetype
    }
    /// Take the new points this frame.
    /// # Errors
    /// If there are no points to take and the stroke has ended, an error is
    /// returned describing how the stroke ended. The value of [`Ended`] should
    /// be respected for how this is handled.
    ///
    /// Notably, that means that just because this function doesn't return an
    /// error doesn't mean that the stroke *hasn't* already ended.
    ///
    /// *On error, this listener should be dropped soon.*
    pub fn take_stroke(&'_ mut self) -> Result<StrokeSlice<'_>, Ended> {
        let poses = self.stream.try_iter();
        Ok(collect(poses, self.archetype, &mut self.staging))
    }
    /// Get the latest position of the pointer, in logical pixels.
    pub fn latest_position(&self) -> ultraviolet::Vec2 {
        todo!()
    }
    /// `Some` if the stroke has ended. You should respect the value of
    /// [`Ended`] to decided whether the action of this stroke should be ignored
    /// or committed.
    ///
    /// This may return `Some` even if there are pending points that can be
    /// acquired using `take_stroke`.
    pub fn is_ended(&self) -> Option<Ended> {
        todo!()
    }
}
trait StylusAxes {
    fn get_axis(&self, axis: StylusAxis) -> Option<f32>;
    fn set_axis(&mut self, axis: StylusAxis, value: f32) -> Result<(), ()>;
    fn has_axis(&self, axis: StylusAxis) -> bool {
        self.get_axis(axis).is_some()
    }
}
#[derive(Debug, Clone, Copy)]
pub struct StylusEvent {
    pub pos: (f32, f32),
    pub pressed: bool,
    pub pressure: Option<f32>,
    pub tilt: Option<(f32, f32)>,
    pub dist: Option<f32>,
}
impl StylusEvent {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            pos: (0.0, 0.0),
            pressed: false,
            pressure: None,
            tilt: None,
            dist: None,
        }
    }
}
impl StylusAxes for StylusEvent {
    fn get_axis(&self, axis: StylusAxis) -> Option<f32> {
        match axis {
            StylusAxis::Dist => self.dist,
            StylusAxis::PosX => Some(self.pos.0),
            StylusAxis::PosY => Some(self.pos.1),
            StylusAxis::Pressure => self.pressure,
            StylusAxis::TiltX => self.tilt.map(|tilt| tilt.0),
            StylusAxis::TiltY => self.tilt.map(|tilt| tilt.1),
        }
    }
    fn set_axis(&mut self, axis: StylusAxis, value: f32) -> Result<(), ()> {
        match axis {
            StylusAxis::Dist => self.dist = Some(value),
            StylusAxis::PosX => self.pos.0 = value,
            StylusAxis::PosY => self.pos.1 = value,
            StylusAxis::Pressure => self.pressure = Some(value),
            StylusAxis::TiltX => {
                let tilt_y = self.tilt.unwrap_or_default().1;
                self.tilt = Some((value, tilt_y));
            }
            StylusAxis::TiltY => {
                let tilt_x = self.tilt.unwrap_or_default().0;
                self.tilt = Some((tilt_x, value));
            }
        }
        Ok(())
    }
}

pub struct WinitStylusEventCollector {
    mouse_pressed: bool,
    pressure: Option<f32>,
    events: Vec<StylusEvent>,

    frame_channel: tokio::sync::broadcast::Sender<StylusEventFrame>,
}
impl Default for WinitStylusEventCollector {
    fn default() -> Self {
        let (sender, _) = tokio::sync::broadcast::channel(32);
        Self {
            mouse_pressed: false,
            events: Vec::new(),
            frame_channel: sender,
            pressure: None,
        }
    }
}
impl WinitStylusEventCollector {
    pub fn push_position(&mut self, pos: (f32, f32)) {
        let event = StylusEvent {
            pos,
            pressed: self.mouse_pressed,
            pressure: Some(
                self.pressure
                    .unwrap_or(if self.mouse_pressed { 1.0 } else { 0.0 }),
            ),
            ..StylusEvent::empty()
        };

        self.pressure = None;

        self.events.push(event);
    }
    pub fn set_pressure(&mut self, pressure: f32) {
        self.pressure = Some(pressure);
    }
    pub fn set_mouse_pressed(&mut self, pressed: bool) {
        self.mouse_pressed = pressed;
        if !pressed {
            self.pressure = None;
        }
    }
    /// This frame is complete, and no more axis events will occur until next frame.
    /// Finish the current event.
    pub fn finish(&mut self) {
        // Notify listeners
        self.broadcast();
    }
    /// Consume the events for this frame, and broadcast them to all listeners.
    /// Events will be accumulated in the case of no listeners.
    fn broadcast(&mut self) {
        let inner_frame = self.take_frame();
        let frame = StylusEventFrame(std::sync::Arc::new(inner_frame));
        if let Err(err) = self.frame_channel.send(frame) {
            //The frame failed to send, recover it!
            //The only reference is stored in err.0.0, thus
            //into_inner will never fail.
            let inner_frame = std::sync::Arc::into_inner(err.0.0).unwrap();

            self.recover_frame(inner_frame);
        }
    }
    #[must_use]
    pub fn frame_receiver(&self) -> tokio::sync::broadcast::Receiver<StylusEventFrame> {
        self.frame_channel.subscribe()
    }
    /// Take all the data and construct a frame from it for broadcast.
    fn take_frame(&mut self) -> StylusEventFrameInner {
        StylusEventFrameInner {
            events: std::mem::take(&mut self.events),
        }
    }
    /// Take a frame and repopulate self. Useful for failed broadcasts.
    fn recover_frame(&mut self, frame: StylusEventFrameInner) {
        self.events = frame.events;
    }
}

pub struct StylusEventFrameInner {
    events: Vec<StylusEvent>,
}

#[derive(Clone)]
pub struct StylusEventFrame(std::sync::Arc<StylusEventFrameInner>);

impl std::ops::Deref for StylusEventFrame {
    type Target = [StylusEvent];
    fn deref(&'_ self) -> &'_ Self::Target {
        &self.0.events
    }
}
