"""Colors and the Qt style sheet of the app."""

BACKGROUND = "#f3f0e9"
PANEL = "#ffffff"
TEXT = "#23272e"
MUTED = "#6b7280"
BORDER = "#d8d2c4"
HEADER = "#1f2a37"
ACCENT = "#d97706"
PRIMARY = "#0f766e"
PRIMARY_DARK = "#115e59"

# One color per verdict, used for badges, the chart and the report.
VERDICT_COLORS = {
    "consistent": "#067647",
    "partially_consistent": "#b54708",
    "contradictory": "#b42318",
}
STATUS_COLORS = {
    "supported": "#067647",
    "agrees": "#067647",
    "contradicted": "#b42318",
    "conflicts": "#b42318",
    "not_mentioned": "#6b7280",
    "new_detail": "#6b7280",
}

STYLE_SHEET = f"""
QWidget {{
    background: {BACKGROUND};
    color: {TEXT};
    font-family: "Segoe UI";
    font-size: 10pt;
}}
QFrame#header {{
    background: {HEADER};
    border-bottom: 4px solid {ACCENT};
}}
QLabel#title {{
    background: transparent;
    color: white;
    font-size: 16pt;
    font-weight: bold;
}}
QLabel#subtitle {{
    background: transparent;
    color: #cbd5e1;
}}
QLabel#section {{
    color: {HEADER};
    font-weight: bold;
}}
QTextEdit, QPlainTextEdit, QTextBrowser, QLineEdit, QListWidget, QTableWidget, QComboBox, QSpinBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px;
    selection-background-color: {PRIMARY};
    selection-color: white;
}}
QPushButton {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 12px;
}}
QPushButton:hover {{
    border-color: {PRIMARY};
    color: {PRIMARY};
}}
QPushButton#primary {{
    background: {PRIMARY};
    border-color: {PRIMARY};
    color: white;
    font-weight: bold;
}}
QPushButton#primary:hover {{
    background: {PRIMARY_DARK};
    color: white;
}}
QPushButton:disabled {{
    background: #e5e1d8;
    color: {MUTED};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {PANEL};
}}
QTabBar::tab {{
    background: #e7e2d6;
    border: 1px solid {BORDER};
    padding: 7px 16px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {PANEL};
    border-top: 3px solid {ACCENT};
    font-weight: bold;
}}
QHeaderView::section {{
    background: {HEADER};
    color: white;
    padding: 4px;
    border: none;
}}
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {PANEL};
    text-align: center;
}}
QProgressBar::chunk {{
    background: {PRIMARY};
}}
"""


def badge(text, color):
    """A small colored label for HTML text boxes."""
    return (f'<span style="background:{color}; color:white; font-weight:bold;">'
            f'&nbsp;{text}&nbsp;</span>')
