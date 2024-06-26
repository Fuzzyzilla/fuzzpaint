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

/// Emulate pointer input from octotablet. This is stateless - that is, the function doesn't remember states of buttons or what is the "current"
/// device. All buttons other than the nib are reported as right-clicks.
/// # Safety
/// Any winit event that reports a device id will use the `dummy` ID. See [`winit::event::DeviceId`] for safe treatment of this special Id.
#[must_use]
pub unsafe fn winit_event_from_octotablet(
    event: &octotablet::events::ToolEvent,
    scale_factor: f64,
) -> Option<winit::event::WindowEvent> {
    use octotablet::events::ToolEvent;
    use winit::event::{DeviceId, ElementState, WindowEvent};

    // Safety: forwarded to this fn's contract. Any user of this return value must be sure to not use
    // the returned event for evil.
    let device_id = unsafe { DeviceId::dummy() };

    Some(match event {
        ToolEvent::Down => WindowEvent::MouseInput {
            device_id,
            state: ElementState::Pressed,
            button: winit::event::MouseButton::Left,
        },
        ToolEvent::Up => WindowEvent::MouseInput {
            device_id,
            state: ElementState::Released,
            button: winit::event::MouseButton::Left,
        },
        ToolEvent::Button { pressed, .. } => WindowEvent::MouseInput {
            device_id,
            state: if *pressed {
                ElementState::Pressed
            } else {
                ElementState::Released
            },
            button: winit::event::MouseButton::Right,
        },
        ToolEvent::Pose(octotablet::axis::Pose { position, .. }) => WindowEvent::CursorMoved {
            device_id,
            position: winit::dpi::PhysicalPosition::from_logical(
                winit::dpi::LogicalPosition::new(position[0], position[1]),
                scale_factor,
            ),
        },

        ToolEvent::Out => WindowEvent::CursorLeft { device_id },
        _ => return None,
    })
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
            let inner_frame = std::sync::Arc::into_inner(err.0 .0).unwrap();

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
