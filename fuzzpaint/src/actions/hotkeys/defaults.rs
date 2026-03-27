use super::{super::Action, KeyboardHotkey, Modifiers};
use winit::keyboard::KeyCode;

pub const KEYBOARD: &[(Action, &[KeyboardHotkey])] = &[
    (
        Action::New,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::KeyN,
        }],
    ),
    (
        Action::NewFromClipboard,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL_SHIFT,
            key: KeyCode::KeyN,
        }],
    ),
    (
        Action::Close,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::KeyW,
        }],
    ),
    (
        Action::Undo,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::KeyZ,
        }],
    ),
    (
        Action::Redo,
        &[
            KeyboardHotkey {
                modifiers: Modifiers::CTRL,
                key: KeyCode::KeyY,
            },
            KeyboardHotkey {
                modifiers: Modifiers::CTRL_SHIFT,
                key: KeyCode::KeyZ,
            },
        ],
    ),
    (
        Action::ViewportPan,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::Space,
        }],
    ),
    (
        Action::ViewportScrub,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::Space,
        }],
    ),
    (
        Action::ViewportRotate,
        &[KeyboardHotkey {
            modifiers: Modifiers::SHIFT,
            key: KeyCode::Space,
        }],
    ),
    (
        Action::ViewportFlipHorizontal,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::KeyM,
        }],
    ),
    (
        Action::ZoomIn,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::Equal,
        }],
    ),
    (
        Action::ZoomOut,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::Minus,
        }],
    ),
    (
        Action::Picker,
        &[
            KeyboardHotkey {
                // Just left control. This looks weird~. putting CTRL in the
                // modifiers would work, but then it would give it a higher
                // priority than we want.
                modifiers: Modifiers::empty(),
                key: KeyCode::ControlLeft,
            },
            KeyboardHotkey {
                modifiers: Modifiers::empty(),
                key: KeyCode::KeyI,
            },
        ],
    ),
    (
        Action::Brush,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::KeyB,
        }],
    ),
    (
        Action::Erase,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::KeyE,
        }],
    ),
    (
        Action::Lasso,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::KeyL,
        }],
    ),
    (
        Action::BrushSizeDown,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::BracketLeft,
        }],
    ),
    (
        Action::BrushSizeUp,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::BracketRight,
        }],
    ),
    (
        Action::ColorSwap,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::KeyX,
        }],
    ),
    (
        Action::Marquee,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::KeyR,
        }],
    ),
    (
        Action::LayerDelete,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::Delete,
        }],
    ),
    (
        Action::LayerUp,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::ArrowUp,
        }],
    ),
    (
        Action::LayerDown,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::ArrowDown,
        }],
    ),
    (
        Action::Text,
        &[KeyboardHotkey {
            modifiers: Modifiers::empty(),
            key: KeyCode::KeyT,
        }],
    ),
    (
        Action::Transform,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL,
            key: KeyCode::KeyT,
        }],
    ),
    (
        Action::FreeTranform,
        &[KeyboardHotkey {
            modifiers: Modifiers::CTRL_SHIFT,
            key: KeyCode::KeyT,
        }],
    ),
];
