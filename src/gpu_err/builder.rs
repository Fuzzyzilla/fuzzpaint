use super::*;
pub struct GpuResultBuilder<OkTy, ErrTy: Into<GpuErrorSource>> {
    result: std::result::Result<OkTy, ErrTy>,
    fatal_if_err: bool,
}
impl<OkTy, ErrTy: Into<GpuErrorSource>> GpuResultBuilder<OkTy, ErrTy> {
    pub fn new_fatal(res: std::result::Result<OkTy, ErrTy>) -> Self {
        Self {
            result: res,
            fatal_if_err: true,
        }
    }
    pub fn new_recoverable(res: std::result::Result<OkTy, ErrTy>) -> Self {
        Self {
            result: res,
            fatal_if_err: false,
        }
    }
    pub fn result(self) -> super::GpuResult<OkTy> {
        match self.result {
            Ok(ok) => Ok(ok),
            Err(err) => Err(GpuError {
                fatal: self.fatal_if_err,
                source: err.into(),
            }),
        }
    }
}

impl<OkyTy, ErrTy> HasDeviceLoss for GpuResultBuilder<OkyTy, ErrTy>
where
    ErrTy: Into<GpuErrorSource> + HasDeviceLoss,
{
    fn device_lost(&self) -> bool {
        self.result
            .as_ref()
            .err()
            .is_some_and(|err| err.device_lost())
    }
}
impl<OkTy, ErrTy> HasOom for GpuResultBuilder<OkTy, ErrTy>
where
    ErrTy: Into<GpuErrorSource> + HasOom,
{
    fn oom(&self) -> Option<vulkano::OomError> {
        self.result
            .as_ref()
            .err()
            .map(|err| err.oom())
            .unwrap_or(None)
    }
}
impl<OkTy, ErrTy> GpuResultInspection for GpuResultBuilder<OkTy, ErrTy>
where
    ErrTy: Into<GpuErrorSource> + Clone,
{
    type ErrTy = ErrTy;
    fn unless(self, p: impl FnOnce(Self::ErrTy) -> bool) -> Self {
        match self.result {
            Ok(..) => self,
            Err(err) => {
                let unless = p(err.clone());

                let new_fatal = self.fatal_if_err ^ unless;

                Self {
                    fatal_if_err: new_fatal,
                    result: Err(err),
                }
            }
        }
    }
}
