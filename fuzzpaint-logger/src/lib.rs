#[derive(Clone)]
pub struct Record {
    pub level: log::Level,
    pub time: time::UtcDateTime,
    pub text: String,
}
impl Record {
    fn from_log(from: &log::Record) -> Self {
        Self {
            level: from.level(),
            time: time::UtcDateTime::now(),
            text: from.args().to_string(),
        }
    }
}
pub struct RecordSet<'a> {
    vec: Vec<Record>,
    _logger: &'a CollectLogger,
}
impl<'a> IntoIterator for RecordSet<'a> {
    type IntoIter = std::vec::IntoIter<Record>;
    type Item = Record;
    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}
pub struct CollectLogger {
    inner: parking_lot::Mutex<Inner>,
    tee: Option<&'static dyn log::Log>,
    filter: log::LevelFilter,
}
impl Default for CollectLogger {
    fn default() -> Self {
        Self {
            inner: Default::default(),
            tee: None,
            filter: log::LevelFilter::Debug,
        }
    }
}
#[derive(Default)]
struct Inner {
    vec: Vec<Record>,
}
impl CollectLogger {
    pub fn new() -> Self {
        Self::default()
    }
    /// Forward messages to this other logger. The other logger's filters will
    /// dictate this logger's filtering.
    pub fn with_tee(self, tee: &'static dyn log::Log) -> Self {
        Self {
            tee: Some(tee),
            ..self
        }
    }
    pub fn with_level(self, level: log::LevelFilter) -> Self {
        Self {
            filter: level,
            ..self
        }
    }
    pub fn install<'a>(self) -> Result<&'a Self, log::SetLoggerError> {
        log::set_max_level(self.filter);
        let this = &*Box::leak(Box::new(self));
        log::set_logger(this).map(|()| this)
    }
    /// Consume all of the enqueued records, and return them.
    pub fn take(&'_ self) -> RecordSet<'_> {
        let vec = std::mem::take(&mut self.inner.lock().vec);
        RecordSet { vec, _logger: self }
    }
}
impl log::Log for CollectLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() > self.filter || self.tee.is_some_and(|tee| tee.enabled(metadata))
    }
    fn flush(&self) {}
    fn log(&self, record: &log::Record) {
        if let Some(tee) = self.tee {
            tee.log(record);
            // Inherit Filters from the tee.
            if !tee.enabled(record.metadata()) {
                return;
            }
        }
        let record = Record::from_log(record);
        self.inner.lock().vec.push(record);
    }
}
