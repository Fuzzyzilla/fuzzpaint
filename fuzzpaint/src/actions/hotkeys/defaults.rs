use super::super::Action;
use super::KeyboardHotkey;
use winit::keyboard::KeyCode;

pub const KEYBOARD: &[(Action, &[KeyboardHotkey])] = &[
    (
        Action::Undo,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: KeyCode::KeyZ,
        }],
    ),
    (
        Action::Redo,
        &[
            KeyboardHotkey {
                alt: false,
                ctrl: true,
                shift: false,
                key: KeyCode::KeyY,
            },
            KeyboardHotkey {
                alt: false,
                ctrl: true,
                shift: true,
                key: KeyCode::KeyZ,
            },
        ],
    ),
    (
        Action::ViewportPan,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::Space,
        }],
    ),
    (
        Action::ViewportScrub,
        &[
            KeyboardHotkey {
                alt: false,
                ctrl: true,
                shift: false,
                key: KeyCode::Space,
            },
            // FIXME: shadowing bug means the true hotkey^^^ is unusable :V
            KeyboardHotkey {
                alt: false,
                ctrl: false,
                shift: false,
                key: KeyCode::KeyS,
            },
        ],
    ),
    (
        Action::ViewportRotate,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::KeyR,
        }],
    ),
    (
        Action::ViewportFlipHorizontal,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::KeyM,
        }],
    ),
    (
        Action::ZoomIn,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: KeyCode::Equal,
        }],
    ),
    (
        Action::ZoomOut,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: KeyCode::Minus,
        }],
    ),
    (
        Action::Picker,
        &[KeyboardHotkey {
            // FIXME: Should be just ctrl, but not possible yet.
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::KeyI,
        }],
    ),
    (
        Action::Gizmo,
        &[KeyboardHotkey {
            // FIXME: Should be just ctrl, but not possible yet.
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::KeyG,
        }],
    ),
    (
        Action::Brush,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::KeyB,
        }],
    ),
    (
        Action::Erase,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::KeyE,
        }],
    ),
    (
        Action::Lasso,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::KeyL,
        }],
    ),
    (
        Action::BrushSizeDown,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::BracketLeft,
        }],
    ),
    (
        Action::BrushSizeUp,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::BracketRight,
        }],
    ),
    (
        Action::Lasso,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::KeyL,
        }],
    ),
    (
        Action::LayerNew,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: KeyCode::KeyT,
        }],
    ),
    (
        Action::LayerDelete,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: KeyCode::Delete,
        }],
    ),
    (
        Action::LayerUp,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: KeyCode::ArrowUp,
        }],
    ),
    (
        Action::LayerDown,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: KeyCode::ArrowDown,
        }],
    ),
];
