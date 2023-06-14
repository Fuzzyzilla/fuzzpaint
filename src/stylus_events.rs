use std::{future::Future, sync::Arc};


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
trait StylusAxes {
    fn get_axis(&self, axis: StylusAxis) -> Option<f32>;
    fn set_axis(&mut self, axis: StylusAxis, value: f32) -> Result<(), ()>;
    fn has_axis(&self, axis: StylusAxis) -> bool {
        self.get_axis(axis).is_some()
    }
}
pub struct StylusEvent {
    pos: (f32, f32),
    pressed: bool,
    pressure: Option<f32>,
    tilt: Option<(f32, f32)>,
    dist: Option<f32>,
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
            _ => None,
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
            _ => return Err(()),
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy)]
pub struct PartialStylusEvent {
    pos_x: Option<f32>,
    pos_y: Option<f32>,
    pressed: Option<bool>,
    pressure: Option<f32>,
    tilt_x: Option<f32>,
    tilt_y: Option<f32>,
    dist: Option<f32>,
}
impl PartialStylusEvent {
    pub fn none() -> Self {
        Self {
            dist: None,
            pos_x: None,
            pos_y: None,
            pressed: None,
            pressure: None,
            tilt_x: None,
            tilt_y: None,
        }
    }
    pub fn is_pos_complete(&self) -> bool {
        self.pos_x.is_some() && self.pos_y.is_some()
    }
}
impl StylusAxes for PartialStylusEvent {
    fn get_axis(&self, axis: StylusAxis) -> Option<f32> {
        match axis {
            StylusAxis::Dist => self.dist,
            StylusAxis::PosX => self.pos_x,
            StylusAxis::PosY => self.pos_y,
            StylusAxis::Pressure => self.pressure,
            StylusAxis::TiltX => self.tilt_x,
            StylusAxis::TiltY => self.tilt_y,
            _ => None,
        }
    }
    fn set_axis(&mut self, axis: StylusAxis, value: f32) -> Result<(), ()> {
        match axis {
            StylusAxis::Dist => self.dist = Some(value),
            StylusAxis::PosX => self.pos_x = Some(value),
            StylusAxis::PosY => self.pos_y = Some(value),
            StylusAxis::Pressure => self.pressure = Some(value),
            StylusAxis::TiltX => self.tilt_x = Some(value),
            StylusAxis::TiltY => self.tilt_y = Some(value),
            _ => return Err(()),
        }
        Ok(())
    }
}
impl TryFrom<PartialStylusEvent> for StylusEvent {
    //There's only one error, and that is the event is incomplete.
    type Error = ();
    fn try_from(value: PartialStylusEvent) -> Result<Self, Self::Error> {
        //Required fields - short circuit None to Err(())
        let pos = value.pos_x.zip(value.pos_y).ok_or(())?;
        let pressed = value.pressed.unwrap_or(false);

        // None if both none, Some with defaults if either or both are set.
        // Feels like there should be a more concise way :P
        let tilt = if value.tilt_x.is_some() || value.tilt_y.is_some() {
            let tilt_x = value.tilt_x.unwrap_or(0.0);
            let tilt_y = value.tilt_y.unwrap_or(0.0);
            Some((tilt_x, tilt_y))
        } else {
            None
        };

        Ok(Self{
            pos,
            pressed,
            pressure: value.pressure,
            tilt,
            dist: value.dist
        })
    }
}


pub struct WinitStylusEventCollector {
    cur_event: Option<PartialStylusEvent>,
    events: Vec<StylusEvent>,

    frame_channel: tokio::sync::broadcast::Sender<StylusEventFrame>,
}
impl WinitStylusEventCollector {
    pub fn push_position(&mut self, pos: (f32, f32)) {
        if let Some(mut event) = self.cur_event.take() {
            // We are adding a position, but the previous already has one!
            // must be a new event.
            if event.is_pos_complete() {
                if let Ok(event) = event.try_into() {
                    self.events.push(event)
                } else {
                    //Failed to complete event. Will fallthrough to
                    //create a new one from this pos.
                    log::warn!("Incomplete event pushed: {event:?}");
                }
            } else {
                //Add pos and return uwu
                event.pos_x = Some(pos.0);
                event.pos_y = Some(pos.1);

                self.cur_event = Some(event);
                return;
            }
        }
        // There was no current event, build a new one.
        let event = PartialStylusEvent {
            pos_x: Some(pos.0),
            pos_y: Some(pos.1),
            ..PartialStylusEvent::none()
        };
        self.cur_event = Some(event);
    }
    pub fn push_axis(&mut self, axis: StylusAxis, value: f32) {
        if let Some(mut event) = self.cur_event.take() {
            //Trying to add this axis, but it already had it- must be a new event!
            if event.has_axis(axis) {
                if let Ok(event) = event.try_into() {
                    self.events.push(event)
                } else {
                    //Failed to complete event. Will fallthrough to
                    //create a new one from this axis.
                    log::warn!("Incomplete event pushed: {event:?}");
                }
            } else {
                //Add and return
                //Dont care if invalid axis.
                let _ = event.set_axis(axis, value);
                self.cur_event = Some(event);
            }
        }
        let mut event = PartialStylusEvent::none();
        //Dont care if invalid axis.
        let _ = event.set_axis(axis, value);
        self.cur_event = Some(event);
    }
    /// This frame is complete, and no more axis events will occur until next frame.
    /// Finish the current event.
    pub fn finish(&mut self) {
        //Take the event-in-progress
        let Some(event) = self.cur_event.take() else {return};

        //finish it!
        if let Ok(event) = event.try_into() {
            self.events.push(event);
        } else {
            log::warn!("Incomplete stylus event at end-of-frame! {event:?}");
        }

        //Notify listeners
        self.broadcast()
    }
    /// Consume the events for this frame, and broadcast them to all listeners.
    /// Events will be accumulated in the case of no listeners.
    fn broadcast(&mut self) {
        let inner_frame = self.take_frame();
        let frame = StylusEventFrame(Arc::new(inner_frame));
        if let Err(err) = self.frame_channel.send(frame) {
            //The frame failed to send, recover it!
            //The only reference is stored in err.0.0, thus
            //into_inner will never fail.
            let inner_frame = Arc::into_inner(err.0.0).unwrap();

            self.recover_frame(inner_frame);
        }
    }
    pub fn frame_receiver(&self) -> tokio::sync::broadcast::Receiver<StylusEventFrame> {
        self.frame_channel.subscribe()
    }
    /// Take all the data and construct a frame from it for broadcast.
    fn take_frame(&mut self) -> StylusEventFrameInner {
        StylusEventFrameInner {
            events: std::mem::take(&mut self.events)
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