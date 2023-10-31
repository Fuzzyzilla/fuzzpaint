//! # Brush

pub type BrushID = crate::id::FuzzID<Brush>;

#[derive(PartialEq, Eq, Hash, strum::AsRefStr, strum::EnumIter, Copy, Clone)]
pub enum BrushKind {
    Stamped,
    Rolled,
}
pub enum BrushStyle {
    Stamped { spacing: f32 },
    Rolled,
}
impl BrushStyle {
    pub fn default_for(brush_kind: BrushKind) -> Self {
        match brush_kind {
            BrushKind::Stamped => Self::Stamped { spacing: 2.0 },
            BrushKind::Rolled => Self::Rolled,
        }
    }
    pub fn brush_kind(&self) -> BrushKind {
        match self {
            Self::Stamped { .. } => BrushKind::Stamped,
            Self::Rolled => BrushKind::Rolled,
        }
    }
}
impl Default for BrushStyle {
    fn default() -> Self {
        Self::default_for(BrushKind::Stamped)
    }
}
pub struct Brush {
    name: String,

    style: BrushStyle,

    id: crate::FuzzID<Brush>,

    //Globally unique ID, for allowing files to be shared after serialization
    universal_id: uuid::Uuid,
}
impl Brush {
    pub fn style(&self) -> &BrushStyle {
        &self.style
    }
    pub fn style_mut(&mut self) -> &mut BrushStyle {
        &mut self.style
    }
    pub fn id(&self) -> BrushID {
        self.id
    }
    pub fn universal_id(&self) -> &uuid::Uuid {
        &self.universal_id
    }
    pub fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }
}
impl Default for Brush {
    fn default() -> Self {
        let id = crate::FuzzID::default();
        Self {
            name: format!("Brush {}", id.id()),
            style: Default::default(),
            id,
            universal_id: uuid::Uuid::new_v4(),
        }
    }
}

pub fn todo_brush() -> Brush {
    static TODO_ID: std::sync::OnceLock<BrushID> = std::sync::OnceLock::new();
    Brush {
        name: "Todo".into(),
        style: Default::default(),
        id: *TODO_ID.get_or_init(Default::default),
        // Example UUID from wikipedia lol
        universal_id: uuid::uuid!("123e4567-e89b-12d3-a456-426614174000"),
    }
}
