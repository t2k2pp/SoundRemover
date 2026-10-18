"""
SoundRemover デスクトップGUI用 モダンダークスタジオ・スタイルシート (`theme.py`)
"""

STUDIO_DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #181a20;
    color: #e6e9ef;
    font-family: "Segoe UI", "Yu Gothic UI", "Meiryo", sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #2d323e;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #9ecbff;
    background-color: #1f222a;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
}

QTabWidget::pane {
    border: 1px solid #2d323e;
    border-radius: 6px;
    background-color: #1e2129;
}

QTabBar::tab {
    background-color: #16181d;
    color: #9aa2b1;
    border: 1px solid #2d323e;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 14px;
    margin-right: 3px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #262b36;
    color: #ffffff;
    border-top: 2px solid #3b82f6;
}

QTabBar::tab:hover {
    background-color: #222630;
    color: #d0d7e2;
}

QPushButton {
    background-color: #2b313e;
    color: #f0f4f8;
    border: 1px solid #3d4556;
    border-radius: 5px;
    padding: 6px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #363e4f;
    border-color: #525d73;
}

QPushButton:pressed {
    background-color: #232834;
}

QPushButton:disabled {
    background-color: #1c1f26;
    color: #5c6370;
    border-color: #292d38;
}

QPushButton#primaryButton {
    background-color: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
    font-size: 14px;
    padding: 9px 16px;
}

QPushButton#primaryButton:hover {
    background-color: #1d4ed8;
}

QPushButton#successButton {
    background-color: #059669;
    border-color: #10b981;
    color: #ffffff;
    font-size: 14px;
    padding: 9px 16px;
}

QPushButton#successButton:hover {
    background-color: #047857;
}

QPushButton#accentButton {
    background-color: #7c3aed;
    border-color: #8b5cf6;
    color: #ffffff;
}

QPushButton#accentButton:hover {
    background-color: #6d28d9;
}

QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background-color: #14161b;
    color: #e6e9ef;
    border: 1px solid #383f4f;
    border-radius: 4px;
    padding: 5px 8px;
}

QComboBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #4f8cff;
}

QListWidget, QTableWidget {
    background-color: #14161b;
    border: 1px solid #2d323e;
    border-radius: 5px;
    color: #e6e9ef;
}

QListWidget::item {
    padding: 5px;
    border-bottom: 1px solid #22252e;
}

QListWidget::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}

QSlider::groove:horizontal {
    border: 1px solid #2d323e;
    height: 6px;
    background: #14161b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #9ecbff;
    border: 1px solid #2563eb;
    width: 15px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QProgressBar {
    border: 1px solid #2d323e;
    border-radius: 5px;
    background-color: #14161b;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 4px;
}
"""
