use super::super::Action;
use super::KeyboardHotkey;
use winit::event::VirtualKeyCode;

pub const KEYBOARD: &[(Action, &[KeyboardHotkey])] = &[
    (
        Action::Undo,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: VirtualKeyCode::Z,
        }],
    ),
    (
        Action::Redo,
        &[
            KeyboardHotkey {
                alt: false,
                ctrl: true,
                shift: false,
                key: VirtualKeyCode::Y,
            },
            KeyboardHotkey {
                alt: false,
                ctrl: true,
                shift: true,
                key: VirtualKeyCode::Z,
            },
        ],
    ),
    (
        Action::ViewportPan,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::Space,
        }],
    ),
    (
        Action::ViewportScrub,
        &[
            KeyboardHotkey {
                alt: false,
                ctrl: true,
                shift: false,
                key: VirtualKeyCode::Space,
            },
            // FIXME: shadowing bug means the true hotkey^^^ is unusable :V
            KeyboardHotkey {
                alt: false,
                ctrl: false,
                shift: false,
                key: VirtualKeyCode::S,
            },
        ],
    ),
    (
        Action::ViewportRotate,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::R,
        }],
    ),
    (
        Action::ViewportFlipHorizontal,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::M,
        }],
    ),
    (
        Action::Gizmo,
        &[KeyboardHotkey {
            // FIXME: Should be just ctrl, but not possible yet.
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::G,
        }],
    ),
    (
        Action::Brush,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::B,
        }],
    ),
    (
        Action::Erase,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::E,
        }],
    ),
    (
        Action::Lasso,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::L,
        }],
    ),
    (
        Action::BrushSizeDown,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::LBracket,
        }],
    ),
    (
        Action::BrushSizeUp,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::RBracket,
        }],
    ),
    (
        Action::Lasso,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::L,
        }],
    ),
    (
        Action::LayerNew,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: VirtualKeyCode::T,
        }],
    ),
    (
        Action::LayerDelete,
        &[KeyboardHotkey {
            alt: false,
            ctrl: false,
            shift: false,
            key: VirtualKeyCode::Delete,
        }],
    ),
    (
        Action::LayerUp,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: VirtualKeyCode::Up,
        }],
    ),
    (
        Action::LayerDown,
        &[KeyboardHotkey {
            alt: false,
            ctrl: true,
            shift: false,
            key: VirtualKeyCode::Down,
        }],
    ),
];
