"""GUI for interrogation sessions: question-answer turns, manual behavioral observations, analysis and report."""
import webbrowser
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter,
                             QTableWidget, QTextBrowser, QVBoxLayout, QWidget)

import config
from interrogation_analysis import analyze_session
from observations import CHANNELS, INTENSITIES, load_cue_vocabulary, make_observation
from report import render_report, save_report
from sessions import evidence_from_text, evidence_to_text, load_session, new_session, save_session


def section_label(text):
    label = QLabel(text)
    label.setObjectName("section")
    return label


class SessionAnalysisWorker(QThread):
    analysis_ready = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, session, with_emotion, parent=None):
        super().__init__(parent)
        self.session = session
        self.with_emotion = with_emotion

    def run(self):
        try:
            self.analysis_ready.emit(analyze_session(self.session, with_emotion=self.with_emotion,
                                                     progress=lambda f: self.progress.emit(int(f * 100))))
        except Exception as e:  # e.g. a model that cannot be downloaded: report it instead of dying silently
            self.failed.emit(str(e))


class ObservationTable(QTableWidget):
    """One row per manual observation: channel, cue (suggestions or free text), intensity, note."""

    def __init__(self, vocabulary, parent=None):
        super().__init__(0, 4, parent)
        self.vocabulary = vocabulary
        self.setHorizontalHeaderLabels(["Channel", "Observation", "Intensity", "Note"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setColumnWidth(0, 135)
        self.setColumnWidth(1, 185)
        self.setColumnWidth(2, 90)
        self.verticalHeader().setVisible(False)

    def add_observation(self, observation=None):
        row = self.rowCount()
        self.insertRow(row)
        channel_box = QComboBox()
        for key, label in CHANNELS.items():
            channel_box.addItem(label, key)
        cue_box = QComboBox()
        cue_box.setEditable(True)
        intensity_box = QComboBox()
        for value, label in INTENSITIES.items():
            intensity_box.addItem(label, value)
        note_edit = QLineEdit()

        def fill_cues():
            text = cue_box.currentText()
            cue_box.clear()
            cue_box.addItems(self.vocabulary.get(channel_box.currentData(), []))
            cue_box.setEditText(text)

        channel_box.currentIndexChanged.connect(fill_cues)
        if observation:
            channel_box.setCurrentIndex(channel_box.findData(observation["channel"]))
            intensity_box.setCurrentIndex(intensity_box.findData(observation.get("intensity", 2)))
            note_edit.setText(observation.get("note", ""))
        else:
            intensity_box.setCurrentIndex(intensity_box.findData(2))
        fill_cues()
        cue_box.setEditText(observation["cue"] if observation else "")
        for column, widget in enumerate((channel_box, cue_box, intensity_box, note_edit)):
            self.setCellWidget(row, column, widget)

    def set_observations(self, observations):
        self.setRowCount(0)
        for observation in observations:
            self.add_observation(observation)

    def observations(self):
        """Rows with an empty cue are skipped."""
        result = []
        for row in range(self.rowCount()):
            channel, cue, intensity, note = (self.cellWidget(row, column) for column in range(4))
            if cue.currentText().strip():
                result.append(make_observation(channel.currentData(), cue.currentText(), intensity.currentData(),
                                               note.text()))
        return result

    def remove_selected(self):
        rows = sorted({index.row() for index in self.selectedIndexes()} or
                      ({self.currentRow()} if self.currentRow() >= 0 else set()), reverse=True)
        for row in rows:
            self.removeRow(row)


class SessionView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = new_session("session-1", "New interrogation session")
        self.current_turn = -1
        self.analysis = None
        self.worker = None
        self.init_ui()
        self.show_session()

    def init_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        for label, slot in (("New", self.new_session), ("Open…", self.open_session), ("Save…", self.save_session)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            toolbar.addWidget(button)
        toolbar.addWidget(QLabel("Samples:"))
        self.sample_box = QComboBox()
        self.sample_box.addItem("Choose a sample session…", None)
        for path in sorted(config.SESSIONS_DIR.glob("*.json")):
            self.sample_box.addItem(path.stem, path)
        self.sample_box.activated.connect(self.open_sample)
        toolbar.addWidget(self.sample_box)
        toolbar.addStretch()
        self.emotion_check = QCheckBox("Estimate emotion from the words")
        toolbar.addWidget(self.emotion_check)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setObjectName("primary")
        self.analyze_btn.clicked.connect(self.analyze)
        toolbar.addWidget(self.analyze_btn)
        self.export_btn = QPushButton("Export Report…")
        self.export_btn.clicked.connect(self.export_report)
        toolbar.addWidget(self.export_btn)
        layout.addLayout(toolbar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(section_label("Title:"))
        self.title_edit = QLineEdit()
        left_layout.addWidget(self.title_edit)
        left_layout.addWidget(section_label("Person interviewed (name used in the analysis):"))
        self.subject_edit = QLineEdit()
        left_layout.addWidget(self.subject_edit)
        baseline_row = QHBoxLayout()
        baseline_row.addWidget(section_label("Baseline turns (neutral opening questions):"))
        self.baseline_spin = QSpinBox()
        self.baseline_spin.setRange(0, 20)
        baseline_row.addWidget(self.baseline_spin)
        left_layout.addLayout(baseline_row)
        left_layout.addWidget(section_label("Evidence: one item per paragraph, optional [Label] first:"))
        self.evidence_edit = QPlainTextEdit()
        left_layout.addWidget(self.evidence_edit)
        left_layout.addWidget(section_label("Turns:"))
        self.turn_list = QListWidget()
        self.turn_list.currentRowChanged.connect(self.select_turn)
        left_layout.addWidget(self.turn_list)
        turn_buttons = QHBoxLayout()
        for label, slot in (("Add Turn", self.add_turn), ("Delete Turn", self.delete_turn)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            turn_buttons.addWidget(button)
        left_layout.addLayout(turn_buttons)
        splitter.addWidget(left)

        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.addWidget(section_label("Question:"))
        self.question_edit = QPlainTextEdit()
        self.question_edit.setMaximumHeight(80)
        middle_layout.addWidget(self.question_edit)
        middle_layout.addWidget(section_label("Answer:"))
        self.answer_edit = QPlainTextEdit()
        middle_layout.addWidget(self.answer_edit)
        middle_layout.addWidget(section_label("Behavioral observations for this turn (entered by hand):"))
        self.observation_table = ObservationTable(load_cue_vocabulary())
        middle_layout.addWidget(self.observation_table)
        observation_buttons = QHBoxLayout()
        add_observation = QPushButton("Add Observation")
        add_observation.clicked.connect(lambda: self.observation_table.add_observation())
        remove_observation = QPushButton("Remove Observation")
        remove_observation.clicked.connect(self.observation_table.remove_selected)
        observation_buttons.addWidget(add_observation)
        observation_buttons.addWidget(remove_observation)
        middle_layout.addLayout(observation_buttons)
        splitter.addWidget(middle)

        self.report_view = QTextBrowser()
        self.report_view.setOpenExternalLinks(True)
        splitter.addWidget(self.report_view)
        splitter.setSizes([330, 480, 590])
        layout.addWidget(splitter)

    # --- session <-> widgets ---

    def show_session(self):
        """Fill every widget from self.session."""
        self.current_turn = -1
        self.title_edit.setText(self.session["title"])
        self.subject_edit.setText(self.session["subject"])
        self.baseline_spin.setValue(self.session.get("baseline_turns", 0))
        self.evidence_edit.setPlainText(evidence_to_text(self.session["evidence"]))
        self.refresh_turn_list(select=0 if self.session["turns"] else -1)
        self.analysis = None
        self.report_view.setHtml("<p>Enter or open a session, then press <b>Analyze</b>.</p>")

    def refresh_turn_list(self, select):
        self.turn_list.blockSignals(True)
        self.turn_list.clear()
        for i, turn in enumerate(self.session["turns"], start=1):
            preview = " ".join((turn.get("question") or turn["answer"]).split())[:60]
            self.turn_list.addItem(f"{i}. {preview} ({len(turn.get('observations', []))} obs.)")
        self.turn_list.blockSignals(False)
        self.current_turn = -1
        self.turn_list.setCurrentRow(select)
        if select < 0:
            self.select_turn(-1)

    def commit_turn(self):
        """Write the turn editor back into self.session."""
        if 0 <= self.current_turn < len(self.session["turns"]):
            self.session["turns"][self.current_turn] = {
                "question": self.question_edit.toPlainText().strip(),
                "answer": self.answer_edit.toPlainText().strip(),
                "observations": self.observation_table.observations(),
            }

    def commit_session(self):
        self.commit_turn()
        self.session["title"] = self.title_edit.text().strip() or "Untitled session"
        self.session["subject"] = self.subject_edit.text().strip() or "the suspect"
        self.session["baseline_turns"] = self.baseline_spin.value()
        self.session["evidence"] = evidence_from_text(self.evidence_edit.toPlainText())
        self.session["turns"] = [turn for turn in self.session["turns"] if turn["answer"]]

    def select_turn(self, row):
        self.commit_turn()
        self.current_turn = row
        has_turn = 0 <= row < len(self.session["turns"])
        turn = self.session["turns"][row] if has_turn else {"question": "", "answer": "", "observations": []}
        self.question_edit.setPlainText(turn.get("question", ""))
        self.answer_edit.setPlainText(turn["answer"])
        self.observation_table.set_observations(turn.get("observations", []))
        for widget in (self.question_edit, self.answer_edit, self.observation_table):
            widget.setEnabled(has_turn)

    def add_turn(self):
        self.commit_turn()
        self.session["turns"].append({"question": "", "answer": "", "observations": []})
        self.refresh_turn_list(select=len(self.session["turns"]) - 1)
        self.question_edit.setFocus()

    def delete_turn(self):
        row = self.turn_list.currentRow()
        if 0 <= row < len(self.session["turns"]):
            self.current_turn = -1  # the editor content belongs to the deleted turn
            del self.session["turns"][row]
            self.refresh_turn_list(select=min(row, len(self.session["turns"]) - 1))

    # --- files ---

    def load(self, session):
        self.session = session
        self.show_session()

    def new_session(self):
        self.load(new_session(f"session-{datetime.now():%Y%m%d-%H%M%S}", "New interrogation session"))

    def open_session(self):
        config.SESSIONS_DIR.mkdir(exist_ok=True)
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Session", str(config.SESSIONS_DIR), "Session (*.json)")
        if file_name:
            self.open_path(file_name)

    def open_sample(self, index):
        path = self.sample_box.itemData(index)
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            self.load(load_session(path))
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Open Session", f"Could not open the session:\n{e}")

    def save_session(self):
        self.commit_session()
        config.SESSIONS_DIR.mkdir(exist_ok=True)
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Session", str(config.SESSIONS_DIR / f"{self.session['session_id']}.json"), "Session (*.json)")
        if file_name:
            try:
                save_session(self.session, file_name)
            except (OSError, ValueError) as e:
                QMessageBox.critical(self, "Save Session", f"Could not save the session:\n{e}")

    # --- analysis ---

    def analyze(self):
        if self.worker is not None:
            return
        self.commit_session()
        self.refresh_turn_list(select=min(max(self.turn_list.currentRow(), 0), len(self.session["turns"]) - 1))
        if not self.session["turns"]:
            QMessageBox.information(self, "Analyze", "Add at least one turn with an answer first.")
            return
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        # The worker gets its own copy, so editing during the analysis cannot change what is analyzed.
        snapshot = {**self.session, "evidence": [dict(e) for e in self.session["evidence"]],
                    "turns": [{**t, "observations": [dict(o) for o in t["observations"]]}
                              for t in self.session["turns"]]}
        self.worker = SessionAnalysisWorker(snapshot, self.emotion_check.isChecked(), parent=self)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.analysis_ready.connect(self.on_analysis_ready)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_analysis_ready(self, analysis):
        self.analysis = analysis
        self.report_view.setHtml(render_report(analysis))

    def on_analysis_failed(self, message):
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n{message}")

    def export_report(self):
        if self.analysis is None:
            QMessageBox.information(self, "Export Report", "Run the analysis first.")
            return
        config.REPORTS_DIR.mkdir(exist_ok=True)
        default = config.REPORTS_DIR / f"{self.analysis['session']['session_id']}_report.html"
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Report", str(default), "HTML (*.html)")
        if not file_name:
            return
        try:
            save_report(self.analysis, file_name)
        except OSError as e:
            QMessageBox.critical(self, "Export Report", f"Could not write the report:\n{e}")
            return
        webbrowser.open(Path(file_name).as_uri())

    def wait_for_worker(self):
        if self.worker is not None:
            self.worker.wait()
