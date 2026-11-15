import sys
import json
from collections import Counter
from datetime import datetime
from html import escape

import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QLabel, QPushButton,
                             QComboBox, QSplitter, QProgressBar, QFrame,
                             QFileDialog, QTabWidget, QMessageBox,
                             QDialog, QDialogButtonBox, QPlainTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import config
import models
import theme
from data import load_cases, pair_id, split_statements
from nli_consistency import analyze_consistency, summarize_case
from session_view import SessionView
from sessions import session_from_case
from statement_similarity_analysis import analyze_statements

CLAIM_STATUS_TEXT = {
    "supported": "supported",
    "contradicted": "contradicted",
    "not_mentioned": "not mentioned",
    "agrees": "agrees",
    "conflicts": "conflicts",
    "new_detail": "new detail",
}


def verdict_badge(verdict):
    return theme.badge(config.LABEL_DISPLAY[verdict], theme.VERDICT_COLORS[verdict])


def analysis_html(witness):
    """Verdicts and scores of one witness for the Analysis tab."""
    consistency = witness['consistency']
    html = f"<h3>Witness {witness['witness_number']} &nbsp; {verdict_badge(consistency['verdict'])}</h3>"
    html += "<table cellpadding='4'>"
    html += (f"<tr><td>NLI + rules (main result)</td><td>{verdict_badge(consistency['verdict'])}</td>"
             f"<td>strongest contradiction {consistency['contradiction']:.2f}</td></tr>")
    html += (f"<tr><td>NLI only</td><td>{verdict_badge(consistency['nli_verdict'])}</td><td></td></tr>")
    html += (f"<tr><td>Baseline</td><td>{verdict_badge(witness['verdict'])}</td>"
             f"<td>similarity {witness['similarity']:.2f}, key-point overlap {witness['overlap']:.2f}</td></tr>")
    html += "</table>"

    notes = []
    for kind, items in consistency['rule_conflicts'].items():
        for item in items:
            notes.append(f"{kind} conflict: {escape(item)}")
    if consistency['hedges']:
        notes.append("witness is uncertain: " + escape(", ".join(consistency['hedges'])))
    if notes:
        html += "<p>" + "<br>".join(notes) + "</p>"
    return html


def claims_html(witness):
    """Sentence by sentence result of one witness for the Claims tab."""
    consistency = witness['consistency']
    html = f"<h3>Witness {witness['witness_number']} &nbsp; {verdict_badge(consistency['verdict'])}</h3>"
    sections = [("Suspect sentences checked against the witness statement", consistency['suspect_claims']),
                ("Witness sentences checked against the suspect statement", consistency['witness_claims'])]
    for title, claims in sections:
        html += f"<p><b>{title}</b></p><table cellpadding='4'>"
        for claim in claims:
            color = theme.STATUS_COLORS[claim['status']]
            html += (f"<tr><td>{theme.badge(CLAIM_STATUS_TEXT[claim['status']], color)}</td>"
                     f"<td>{escape(claim['claim'])}</td>"
                     f"<td><font color='{theme.MUTED}'>support {claim['support']:.2f}, "
                     f"contradiction {claim['contradiction']:.2f}</font></td></tr>")
        html += "</table>"
    return html


def differences_html(witness):
    """Key points of both statements for the Differences tab."""
    html = f"<h3>Witness {witness['witness_number']} vs suspect</h3><table cellpadding='4'>"
    html += "<tr><th align='left'>Category</th><th align='left'>In both</th>"
    html += "<th align='left'>Only suspect</th><th align='left'>Only witness</th></tr>"
    for category, values in witness['comparison'].items():
        if not values['matching'] and not values['different']:
            continue
        html += (f"<tr><td><b>{category}</b></td>"
                 f"<td><font color='{theme.VERDICT_COLORS['consistent']}'>{escape(', '.join(values['matching']))}"
                 f"</font></td><td>{escape(', '.join(values['suspect_only']))}</td>"
                 f"<td>{escape(', '.join(values['witness_only']))}</td></tr>")
    html += "</table>"
    return html


def summary_html(witnesses):
    html = ""
    for witness in witnesses:
        consistency = witness['consistency']
        contradicted = 0
        for claim in consistency['suspect_claims']:
            if claim['status'] == "contradicted":
                contradicted += 1
        html += (f"<p><b>Witness {witness['witness_number']}</b> &nbsp; {verdict_badge(consistency['verdict'])}<br>"
                 f"{contradicted} of {len(consistency['suspect_claims'])} suspect sentences contradicted</p>")
    return html


class AnalysisWorker(QThread):
    # Named analysis_ready so it does not shadow QThread's own `finished` signal.
    analysis_ready = pyqtSignal(int, dict)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, int)

    def __init__(self, case_index, case, parent=None):
        super().__init__(parent)
        self.case_index = case_index
        self.case = case

    def run(self):
        try:
            self.analysis_ready.emit(self.case_index, self.analyze())
        except Exception as e:  # e.g. a model that cannot be downloaded: report it instead of dying silently
            self.failed.emit(self.case_index, str(e))

    def analyze(self):
        witness_statements = self.case['witness_statements']
        suspect_statement = self.case['suspect_statement']
        total_steps = len(witness_statements) + 1

        witnesses = []
        for i, witness in enumerate(witness_statements, start=1):
            analysis = analyze_statements(suspect_statement, witness)
            analysis['witness_number'] = i
            analysis['witness_statement'] = witness
            analysis['consistency'] = analyze_consistency(suspect_statement, witness)
            witnesses.append(analysis)
            self.progress.emit(self.case_index, int(i * 100 / total_steps))

        doc = models.get_nlp()(suspect_statement + " " + " ".join(witness_statements))
        words = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop:
                words.append(token.text)
        keywords = [word for word, count in Counter(words).most_common(10)]
        self.progress.emit(self.case_index, 100)

        return {'witnesses': witnesses, 'keywords': keywords, 'summary': summarize_case(witnesses)}


class CustomCaseDialog(QDialog):
    """Lets the user paste a suspect statement and witness statements of their own."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analyze Custom Statements")
        self.resize(700, 500)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Suspect statement:"))
        self.suspect_edit = QPlainTextEdit()
        layout.addWidget(self.suspect_edit)
        layout.addWidget(QLabel("Witness statements (separate witnesses with an empty line):"))
        self.witness_edit = QPlainTextEdit()
        layout.addWidget(self.witness_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        if not self.suspect_edit.toPlainText().strip() or not split_statements(self.witness_edit.toPlainText()):
            QMessageBox.warning(self, "Missing text", "Enter a suspect statement and at least one witness statement.")
            return
        super().accept()

    def case(self, case_id):
        suspect = " ".join(self.suspect_edit.toPlainText().split())
        description = "Custom: " + suspect[:50]
        if len(suspect) > 50:
            description += "…"
        return {
            "case_id": case_id,
            "description": description,
            "suspect_statement": suspect,
            "witness_statements": split_statements(self.witness_edit.toPlainText()),
        }


class InterrogationAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InterrogationX")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(theme.STYLE_SHEET)

        self.data = load_cases()
        self.worker = None
        self.requested_index = 0
        self.current_results = None  # (case_index, results) of the case on screen

        self.init_ui()
        self.update_analysis(0)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.build_header())

        # Two views: comparing a suspect statement with witness statements, and full interrogation sessions.
        self.views = QTabWidget()
        self.views.addTab(self.build_comparison_view(), "Statement Comparison")
        self.session_view = SessionView()
        self.views.addTab(self.session_view, "Interrogation Session")
        layout.addWidget(self.views)

    def build_header(self):
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        title = QLabel("InterrogationX")
        title.setObjectName("title")
        subtitle = QLabel("Statement consistency analysis: a helper for the reader, not a lie detector")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
        return header

    def build_comparison_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)

        controls = QHBoxLayout()
        self.case_selector = QComboBox()
        for case in self.data:
            self.case_selector.addItem(f"{case['case_id']} – {case['description']}")
        self.case_selector.currentIndexChanged.connect(self.update_analysis)
        controls.addWidget(self.section_label("Case:"))
        controls.addWidget(self.case_selector, stretch=1)

        self.custom_btn = QPushButton("Analyze Custom Statements…")
        self.custom_btn.setObjectName("primary")
        self.custom_btn.clicked.connect(self.add_custom_case)
        controls.addWidget(self.custom_btn)

        self.session_btn = QPushButton("Open as Session")
        self.session_btn.setToolTip("Continue with this case in the Interrogation Session view: the suspect "
                                    "statement becomes the first answer and the witnesses become the evidence.")
        self.session_btn.clicked.connect(self.open_case_as_session)
        controls.addWidget(self.session_btn)

        self.export_btn = QPushButton("Export Analysis")
        self.export_btn.clicked.connect(self.export_analysis)
        controls.addWidget(self.export_btn)
        layout.addLayout(controls)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.build_left_panel())
        self.result_tabs = self.build_result_tabs()
        splitter.addWidget(self.result_tabs)
        splitter.setSizes([400, 1000])
        layout.addWidget(splitter)
        return view

    def section_label(self, text):
        label = QLabel(text)
        label.setObjectName("section")
        return label

    def read_only_box(self):
        box = QTextEdit()
        box.setReadOnly(True)
        return box

    def build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.section_label("Suspect statement"))
        self.suspect_text = self.read_only_box()
        layout.addWidget(self.suspect_text)
        layout.addWidget(self.section_label("Summary"))
        self.summary_text = self.read_only_box()
        layout.addWidget(self.summary_text)
        return panel

    def build_result_tabs(self):
        tabs = QTabWidget()

        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.addWidget(self.section_label("Witness statements"))
        self.witness_text = self.read_only_box()
        analysis_layout.addWidget(self.witness_text)
        analysis_layout.addWidget(self.section_label("Results"))
        self.analysis_text = self.read_only_box()
        analysis_layout.addWidget(self.analysis_text)
        tabs.addTab(analysis_tab, "Analysis")

        self.claims_text = self.read_only_box()
        tabs.addTab(self.claims_text, "Claims")

        # A plain Figure (not pyplot) keeps matplotlib state local to this widget.
        self.figure = Figure(facecolor=theme.PANEL)
        self.canvas = FigureCanvas(self.figure)
        tabs.addTab(self.canvas, "Scores")

        self.keywords_text = self.read_only_box()
        tabs.addTab(self.keywords_text, "Keywords")

        self.differences_text = self.read_only_box()
        tabs.addTab(self.differences_text, "Differences")
        return tabs

    def add_custom_case(self):
        dialog = CustomCaseDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        custom_count = 0
        for case in self.data:
            if case['case_id'].startswith('custom'):
                custom_count += 1
        case = dialog.case(f"custom{custom_count + 1}")
        self.data.append(case)
        self.case_selector.addItem(f"{case['case_id']} – {case['description']}")
        self.case_selector.setCurrentIndex(len(self.data) - 1)

    def open_case_as_session(self):
        self.session_view.load(session_from_case(self.data[self.case_selector.currentIndex()]))
        self.views.setCurrentWidget(self.session_view)

    def update_analysis(self, index):
        if index < 0:
            return
        self.requested_index = index
        self.current_results = None
        self.suspect_text.setText(self.data[index]['suspect_statement'])
        for box in (self.summary_text, self.witness_text, self.analysis_text, self.claims_text, self.keywords_text,
                    self.differences_text):
            box.setText("Analyzing…")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # One analysis at a time: a running worker starts the latest requested case when it finishes.
        if self.worker is None:
            self.start_worker(index)

    def start_worker(self, index):
        # Parented to the window so Qt, not Python's garbage collector, owns the running thread.
        self.worker = AnalysisWorker(index, self.data[index], parent=self)
        self.worker.progress.connect(self.on_progress)
        self.worker.analysis_ready.connect(self.on_analysis_ready)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        done_index = self.worker.case_index
        self.worker.deleteLater()
        self.worker = None
        if done_index != self.requested_index:
            self.start_worker(self.requested_index)

    def on_progress(self, index, value):
        if index == self.requested_index:
            self.progress_bar.setValue(value)

    def on_analysis_ready(self, index, results):
        if index != self.requested_index:
            return  # stale: the user picked another case while this one was running
        self.current_results = (index, results)
        self.render_results(results)

    def on_analysis_failed(self, index, message):
        if index != self.requested_index:
            return
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n{message}")

    def render_results(self, results):
        witnesses = results['witnesses']
        witness_html = ""
        for witness in witnesses:
            witness_html += (f"<p><b>Witness {witness['witness_number']}</b><br>"
                             f"{escape(witness['witness_statement'])}</p>")

        self.witness_text.setHtml(witness_html)
        self.analysis_text.setHtml("".join(analysis_html(w) for w in witnesses))
        self.claims_text.setHtml("".join(claims_html(w) for w in witnesses))
        self.differences_text.setHtml("".join(differences_html(w) for w in witnesses))
        self.summary_text.setHtml(summary_html(witnesses))
        self.keywords_text.setText("\n".join(results['keywords']))

        self.update_scores_chart(witnesses)
        self.progress_bar.setVisible(False)

    def update_scores_chart(self, witnesses):
        labels = []
        contradiction = []
        similarity = []
        overlap = []
        for witness in witnesses:
            labels.append(f"Witness {witness['witness_number']}")
            contradiction.append(witness['consistency']['contradiction'])
            similarity.append(witness['similarity'])
            overlap.append(witness['overlap'])
        positions = list(range(len(witnesses)))
        width = 0.25

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar([p - width for p in positions], contradiction, width, label='NLI contradiction',
               color=theme.VERDICT_COLORS['contradictory'])
        ax.bar(positions, similarity, width, label='Similarity (e5 cosine)', color=theme.HEADER)
        ax.bar([p + width for p in positions], overlap, width, label='Key-point overlap', color=theme.ACCENT)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Scores per witness')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)
        self.figure.tight_layout()
        self.canvas.draw()

    def export_analysis(self):
        if self.current_results is None:
            QMessageBox.information(self, "Export Analysis", "The analysis of this case is still running.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Analysis", "",
                                                   "JSON Files (*.json);;CSV Files (*.csv)")
        if not file_name:
            return

        index, results = self.current_results
        case = self.data[index]
        rows = []
        claims = {}
        comparisons = {}
        for witness in results['witnesses']:
            pair = pair_id(case['case_id'], witness['witness_number'])
            rows.append({
                'pair_id': pair,
                'case_id': case['case_id'],
                'description': case['description'],
                'witness_index': witness['witness_number'],
                'suspect_statement': case['suspect_statement'],
                'witness_statement': witness['witness_statement'],
                'verdict': witness['consistency']['verdict'],
                'nli_verdict': witness['consistency']['nli_verdict'],
                'baseline_verdict': witness['verdict'],
                'nli_contradiction': witness['consistency']['contradiction'],
                'similarity': witness['similarity'],
                'overlap': witness['overlap'],
            })
            claims[pair] = witness['consistency']
            comparisons[pair] = witness['comparison']

        try:
            if file_name.lower().endswith('.csv'):
                # utf-8-sig so Excel shows quotes and non-ASCII characters correctly.
                pd.DataFrame(rows).to_csv(file_name, index=False, encoding='utf-8-sig')
            else:
                export = {
                    'exported_at': datetime.now().isoformat(timespec='seconds'),
                    'models': {'similarity': config.SIMILARITY_MODEL, 'nli': config.NLI_MODEL},
                    'pairs': rows,
                    'claims': claims,
                    'comparisons': comparisons,
                    'keywords': results['keywords'],
                    'summary': results['summary'],
                }
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(export, f, indent=4, ensure_ascii=False)
        except OSError as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def closeEvent(self, event):
        # Let a running analysis finish so the thread is not destroyed while running.
        if self.worker is not None:
            self.worker.wait()
        self.session_view.wait_for_worker()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InterrogationAnalyzer()
    window.show()
    sys.exit(app.exec())
