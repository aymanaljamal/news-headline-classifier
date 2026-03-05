"""
main_gui.py
واجهة رسومية (GUI) لـ News Headline Classifier
Requires: pip install PyQt5
"""

import sys
import os
import threading
import traceback

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QProgressBar, QFrame, QGridLayout, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush


# ─────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────
BG      = "#1e1e2e"
BG2     = "#2a2a3e"
BG3     = "#12121f"
ACCENT  = "#7c6af7"
ACCENT2 = "#56cfb2"
RED     = "#f7768e"
YELLOW  = "#e0af68"
GREEN   = "#9ece6a"
CYAN    = "#7dcfff"
FG      = "#cdd6f4"
FG_DIM  = "#6c7086"


STYLE = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {FG};
    font-family: 'Segoe UI';
    font-size: 10pt;
}}
QTabWidget::pane {{
    border: 1px solid {BG2};
    background: {BG};
}}
QTabBar::tab {{
    background: {BG2};
    color: {FG_DIM};
    padding: 8px 18px;
    font-weight: bold;
    border: none;
}}
QTabBar::tab:selected {{
    background: {ACCENT};
    color: white;
}}
QTabBar::tab:hover {{
    background: #3a3a5e;
    color: {FG};
}}
QPushButton {{
    background-color: {ACCENT};
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: #9580ff;
}}
QPushButton:disabled {{
    background-color: #3a3a5e;
    color: {FG_DIM};
}}
QPushButton#secondary {{
    background-color: #3a3a5e;
}}
QPushButton#secondary:hover {{
    background-color: #4a4a6e;
}}
QPushButton#danger {{
    background-color: #7a1c1c;
}}
QPushButton#danger:hover {{
    background-color: #a02828;
}}
QLineEdit {{
    background-color: {BG3};
    color: {CYAN};
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    padding: 6px 10px;
    font-family: 'Courier New';
    font-size: 11pt;
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QTextEdit {{
    background-color: {BG3};
    color: {FG};
    border: none;
    font-family: 'Courier New';
    font-size: 10pt;
}}
QTableWidget {{
    background-color: {BG2};
    color: {FG};
    border: none;
    gridline-color: #3a3a5e;
    font-size: 10pt;
}}
QTableWidget::item:selected {{
    background-color: {ACCENT};
}}
QHeaderView::section {{
    background-color: {BG3};
    color: {CYAN};
    padding: 6px;
    border: none;
    font-weight: bold;
}}
QProgressBar {{
    background-color: {BG2};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {ACCENT2};
    border-radius: 4px;
}}
QScrollBar:vertical {{
    background: {BG2};
    width: 8px;
}}
QScrollBar::handle:vertical {{
    background: #3a3a5e;
    border-radius: 4px;
}}
QFrame#section {{
    background-color: {BG2};
    border-radius: 6px;
    border: 1px solid #3a3a5e;
}}
QLabel#title {{
    color: {ACCENT2};
    font-size: 11pt;
    font-weight: bold;
}}
QLabel#header {{
    color: {ACCENT};
    font-size: 16pt;
    font-weight: bold;
}}
"""


# ─────────────────────────────────────────────
#  WORKER THREAD
# ─────────────────────────────────────────────
class PipelineWorker(QObject):
    log_signal     = pyqtSignal(str, str)   # (message, tag)
    step_signal    = pyqtSignal(int, str)   # (index, state: run/ok)
    done_signal    = pyqtSignal(dict)       # results dict
    error_signal   = pyqtSignal(str)

    def __init__(self, data_file, config=None):
        super().__init__()
        self.data_file = data_file
        self.config    = config
        self.classifier = None

    def run(self):
        try:
            self.log_signal.emit("=" * 56 + "\n", "dim")
            self.log_signal.emit(" NEWS CLASSIFIER — PIPELINE START\n", "accent")
            self.log_signal.emit("=" * 56 + "\n", "dim")

            self.log_signal.emit("\nLoading configuration…\n", "cyan")
            from utils.config_loader import load_config
            config, _ = load_config()
            self.config = config
            self.log_signal.emit(f"  {config['app']['name']} v{config['app']['version']}\n", "white")

            from news_classifier.classifier import TextClassifier
            self.classifier = TextClassifier(config)

            data_file = self.data_file or config["data"]["train_file"]
            self.log_signal.emit(f"  Data file: {data_file}\n", "white")

            # Limit to first 10,000 rows
            import pandas as pd, tempfile
            self.log_signal.emit("  Loading first 10,000 rows only...\n", "dim")
            df = pd.read_csv(data_file, nrows=10000)
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
            df.to_csv(tmp.name, index=False)
            tmp.close()
            data_file = tmp.name
            self.log_signal.emit(f"  Rows loaded: {len(df)}\n", "green")

            # Steps 1-2
            self.step_signal.emit(0, "run"); self.step_signal.emit(1, "run")
            self.log_signal.emit("\n[1/6] Preparing data...\n", "yellow")
            self.classifier.prepare_data(data_file)
            self.step_signal.emit(0, "ok"); self.step_signal.emit(1, "ok")
            self.log_signal.emit("  Done.\n", "green")

            # Steps 3-4
            self.step_signal.emit(2, "run"); self.step_signal.emit(3, "run")
            self.log_signal.emit("\n[2/6] Training models…\n", "yellow")
            self.classifier.train_models()
            self.step_signal.emit(2, "ok"); self.step_signal.emit(3, "ok")
            self.log_signal.emit("  Done.\n", "green")

            # Step 5
            self.step_signal.emit(4, "run")
            self.log_signal.emit("\n[3/6] Evaluating…\n", "yellow")
            results = self.classifier.evaluate_models()
            self.step_signal.emit(4, "ok")

            dt = results["decision_tree"]["metrics"]
            lr = results["logistic_regression"]["metrics"]
            for name, m in [("Decision Tree", dt), ("Logistic Regression", lr)]:
                self.log_signal.emit(f"\n  {name}:\n", "white")
                for k, v in m.items():
                    self.log_signal.emit(f"    {k:<12}: {float(v):.4f}\n" if isinstance(v, (int, float)) else f"    {k:<12}: {v}\n", "white")

            # Steps 6-8
            self.step_signal.emit(5, "run"); self.step_signal.emit(6, "run"); self.step_signal.emit(7, "run")
            self.log_signal.emit("\n[4/6] Generating graphs…\n", "yellow")
            os.makedirs("data", exist_ok=True)

            from utils.plot_utils import (plot_model_comparison,
                plot_confusion_matrix, plot_decision_tree_manual)

            plot_model_comparison({"decision_tree": dt, "logistic_regression": lr},
                                  save_path="data/model_comparison.png")
            plot_confusion_matrix(results["decision_tree"]["y_true"],
                                  results["decision_tree"]["y_pred"],
                                  title="Decision Tree - Confusion Matrix",
                                  save_path="data/confusion_dt.png")
            plot_confusion_matrix(results["logistic_regression"]["y_true"],
                                  results["logistic_regression"]["y_pred"],
                                  title="Logistic Regression - Confusion Matrix",
                                  save_path="data/confusion_lr.png")

            feat = getattr(self.classifier, "feature_names", None)
            cls  = config["data"].get("class_names", None)
            plot_decision_tree_manual(self.classifier.decision_tree.root,
                                      feature_names=feat, class_names=cls,
                                      save_path="data/decision_tree_structure.png")

            self.step_signal.emit(5, "ok"); self.step_signal.emit(6, "ok"); self.step_signal.emit(7, "ok")
            self.log_signal.emit("  Graphs saved to data/\n", "green")
            self.log_signal.emit("\n" + "=" * 56 + "\n", "dim")
            self.log_signal.emit(" PIPELINE COMPLETE\n", "green")
            self.log_signal.emit("=" * 56 + "\n", "dim")

            results["_classifier"] = self.classifier
            results["_config"]     = config
            self.done_signal.emit(results)

        except Exception as exc:
            self.error_signal.emit(f"{exc}\n\n{traceback.format_exc()}")


# ─────────────────────────────────────────────
#  BAR CHART WIDGET
# ─────────────────────────────────────────────
class BarChart(QWidget):
    def __init__(self):
        super().__init__()
        self.data = None
        self.setMinimumHeight(240)
        self.setStyleSheet(f"background-color: {BG3}; border-radius: 6px;")

    def set_data(self, dt_metrics, lr_metrics):
        self.data = {"dt": dt_metrics, "lr": lr_metrics}
        self.update()

    def paintEvent(self, event):
        if not self.data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H = self.width(), self.height()
        PL, PR, PT, PB = 55, 20, 20, 45
        pw, ph = W - PL - PR, H - PT - PB

        dt = self.data["dt"]
        lr = self.data["lr"]
        keys   = ["accuracy", "precision", "recall", "f1_score"]
        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        n = len(keys)

        # Grid lines
        p.setPen(QPen(QColor("#2a2a40"), 1, Qt.DashLine))
        for yi in range(0, 11):
            yv = yi / 10
            y  = int(PT + ph - yv * ph)
            p.drawLine(PL, y, PL + pw, y)
            p.setPen(QColor(FG_DIM))
            p.setFont(QFont("Courier New", 8))
            p.drawText(2, y + 4, f"{yv:.1f}")
            p.setPen(QPen(QColor("#2a2a40"), 1, Qt.DashLine))

        gw = pw / n
        bw = int(gw * 0.28)

        for i, (key, lbl) in enumerate(zip(keys, labels)):
            cx = int(PL + gw * (i + 0.5))
            dv = dt.get(key, 0)
            lv = lr.get(key, 0)

            # DT bar
            x0 = cx - bw - 3
            dh = int(dv * ph)
            p.fillRect(x0, PT + ph - dh, bw, dh, QColor(ACCENT))
            p.setPen(QColor(ACCENT))
            p.setFont(QFont("Courier New", 8))
            p.drawText(x0, PT + ph - dh - 4, f"{dv:.3f}")

            # LR bar
            x1 = cx + 3
            lh = int(lv * ph)
            p.fillRect(x1, PT + ph - lh, bw, lh, QColor(ACCENT2))
            p.setPen(QColor(ACCENT2))
            p.drawText(x1, PT + ph - lh - 4, f"{lv:.3f}")

            # X label
            p.setPen(QColor(FG))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(cx - 30, PT + ph + 20, 60, 20, Qt.AlignCenter, lbl)

        # Legend
        p.fillRect(PL + 10, PT + 8, 14, 10, QColor(ACCENT))
        p.setPen(QColor(ACCENT))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(PL + 28, PT + 18, "Decision Tree")

        p.fillRect(PL + 150, PT + 8, 14, 10, QColor(ACCENT2))
        p.setPen(QColor(ACCENT2))
        p.drawText(PL + 168, PT + 18, "Logistic Regression")


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────
class NewsClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("News Headline Classifier")
        self.resize(1200, 840)
        self.setStyleSheet(STYLE)

        self.classifier = None
        self.config_obj = None
        self.worker     = None
        self.thread     = None

        self._build_ui()

    # ──────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setStyleSheet(f"background-color: #13131f;")
        hdr.setFixedHeight(52)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(20, 0, 20, 0)
        lbl = QLabel("📰  News Headline Classifier")
        lbl.setObjectName("header")
        hl.addWidget(lbl)
        hl.addStretch()
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet(f"color: {FG_DIM};")
        hl.addWidget(self.status_lbl)
        main_layout.addWidget(hdr)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs, 1)

        self._build_train_tab()
        self._build_results_tab()
        self._build_predict_tab()
        self._build_log_tab()

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setFixedHeight(8)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

    # ── Train Tab ─────────────────────────────
    def _build_train_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Data file section
        sec = self._section("Data File")
        row = QHBoxLayout()
        row.addWidget(QLabel("CSV / data file:"))
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Leave empty to use config default…")
        row.addWidget(self.file_input, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.setObjectName("secondary")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse)
        row.addWidget(btn_browse)
        sec.layout().addLayout(row)
        layout.addWidget(sec)

        # Pipeline steps
        sec2 = self._section("Pipeline Steps")
        steps = [
            ("1.  Load & clean data",           CYAN),
            ("2.  TF-IDF vectorisation",         CYAN),
            ("3.  Train Decision Tree",          YELLOW),
            ("4.  Train Logistic Regression",    YELLOW),
            ("5.  Evaluate both models",         GREEN),
            ("6.  Generate comparison graph",    GREEN),
            ("7.  Plot confusion matrices",      GREEN),
            ("8.  Plot Decision Tree structure", GREEN),
        ]
        self.step_labels = []
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (txt, col) in enumerate(steps):
            lbl = QLabel(f"  ○  {txt}")
            lbl.setStyleSheet(f"color: {FG_DIM};")
            grid.addWidget(lbl, i // 2, i % 2)
            self.step_labels.append((lbl, col))
        sec2.layout().addLayout(grid)
        layout.addWidget(sec2)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Full Pipeline")
        self.btn_run.setFixedHeight(38)
        self.btn_run.clicked.connect(self._run_pipeline)
        btn_row.addWidget(self.btn_run)

        btn_reset = QPushButton("↺  Reset")
        btn_reset.setObjectName("secondary")
        btn_reset.setFixedWidth(100)
        btn_reset.setFixedHeight(38)
        btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "⚙  Train & Evaluate")

    # ── Results Tab ───────────────────────────
    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Metrics table
        sec = self._section("Metrics Comparison")
        self.metrics_table = QTableWidget(4, 4)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Metric", "Decision Tree", "Logistic Regression", "Winner"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setFixedHeight(160)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.verticalHeader().setVisible(False)
        sec.layout().addWidget(self.metrics_table)
        layout.addWidget(sec)

        # Bar chart
        sec2 = self._section("Visual Comparison")
        self.bar_chart = BarChart()
        sec2.layout().addWidget(self.bar_chart)
        layout.addWidget(sec2)

        # Interpretation
        sec3 = self._section("Interpretation")
        self.interp_text = QTextEdit()
        self.interp_text.setReadOnly(True)
        self.interp_text.setFixedHeight(110)
        sec3.layout().addWidget(self.interp_text)
        layout.addWidget(sec3)

        self.tabs.addTab(tab, "📊  Results")

    # ── Predict Tab ───────────────────────────
    def _build_predict_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Single predict
        sec = self._section("Single Headline Prediction")
        sec.layout().addWidget(QLabel("Enter a headline:"))
        self.hl_input = QLineEdit()
        self.hl_input.setPlaceholderText("e.g. Stock market reaches new high…")
        self.hl_input.returnPressed.connect(self._predict_single)
        sec.layout().addWidget(self.hl_input)

        brow = QHBoxLayout()
        btn_pred = QPushButton("🔍  Predict")
        btn_pred.setFixedWidth(140)
        btn_pred.clicked.connect(self._predict_single)
        brow.addWidget(btn_pred)
        btn_clr = QPushButton("Clear")
        btn_clr.setObjectName("secondary")
        btn_clr.setFixedWidth(80)
        btn_clr.clicked.connect(lambda: self.hl_input.clear())
        brow.addWidget(btn_clr)
        brow.addStretch()
        sec.layout().addLayout(brow)
        layout.addWidget(sec)

        # Result
        sec2 = self._section("Result")
        grid = QGridLayout()
        grid.setSpacing(8)
        self.pred_dt    = QLabel("—")
        self.pred_lr    = QLabel("—")
        self.pred_agree = QLabel("—")
        for i, (lbl_txt, val_lbl) in enumerate([
            ("Decision Tree:",       self.pred_dt),
            ("Logistic Regression:", self.pred_lr),
            ("Agreement:",           self.pred_agree),
        ]):
            lbl = QLabel(lbl_txt)
            lbl.setStyleSheet(f"color: {FG_DIM}; font-weight: bold;")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet(f"color: {CYAN}; font-size: 13pt; font-weight: bold;")
            grid.addWidget(lbl,     i, 0)
            grid.addWidget(val_lbl, i, 1)
        sec2.layout().addLayout(grid)
        layout.addWidget(sec2)

        # Batch
        sec3 = self._section("Batch Test")
        self.batch_table = QTableWidget(0, 3)
        self.batch_table.setHorizontalHeaderLabels(
            ["Headline", "Decision Tree", "Logistic Regression"])
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.batch_table.setFixedHeight(200)
        self.batch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_table.verticalHeader().setVisible(False)
        sec3.layout().addWidget(self.batch_table)
        btn_batch = QPushButton("▶  Run Batch Test")
        btn_batch.setFixedWidth(160)
        btn_batch.clicked.connect(self._run_batch)
        sec3.layout().addWidget(btn_batch)
        layout.addWidget(sec3)

        layout.addStretch()
        self.tabs.addTab(tab, "🔍  Predict")

    # ── Log Tab ───────────────────────────────
    def _build_log_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        brow = QHBoxLayout()
        btn_clr = QPushButton("Clear Log")
        btn_clr.setObjectName("secondary")
        btn_clr.setFixedWidth(110)
        btn_clr.clicked.connect(self._clear_log)
        brow.addWidget(btn_clr)

        btn_save = QPushButton("Save Log…")
        btn_save.setObjectName("secondary")
        btn_save.setFixedWidth(110)
        btn_save.clicked.connect(self._save_log)
        brow.addWidget(btn_save)
        brow.addStretch()
        layout.addLayout(brow)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            f"background:{BG3}; color:{FG}; font-family:'Courier New'; font-size:10pt;")
        layout.addWidget(self.log_box, 1)

        self.tabs.addTab(tab, "📋  Log")

    # ──────────────────────────────────────────
    def _section(self, title):
        frame = QFrame()
        frame.setObjectName("section")
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(12, 10, 12, 10)
        vl.setSpacing(8)
        lbl = QLabel(title)
        lbl.setObjectName("title")
        vl.addWidget(lbl)
        return frame

    # ──────────────────────────────────────────
    #  PIPELINE
    # ──────────────────────────────────────────
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data file", "", "CSV files (*.csv);;All files (*.*)")
        if path:
            self.file_input.setText(path)

    def _run_pipeline(self):
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self._reset_steps()
        self._set_status("Running pipeline…")

        self.worker = PipelineWorker(self.file_input.text().strip())
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log_signal.connect(self._log)
        self.worker.step_signal.connect(self._update_step)
        self.worker.done_signal.connect(self._on_done)
        self.worker.error_signal.connect(self._on_error)
        self.worker.done_signal.connect(self.thread.quit)
        self.worker.error_signal.connect(self.thread.quit)
        self.thread.finished.connect(self._pipeline_finished)

        self.thread.start()

    def _pipeline_finished(self):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

    def _on_done(self, results):
        self.classifier = results.get("_classifier")
        self.config_obj = results.get("_config")
        self._set_status("Pipeline complete ✓")
        self._populate_results(results)
        self.tabs.setCurrentIndex(1)

    def _on_error(self, msg):
        self._log(f"\nERROR:\n{msg}\n", "red")
        self._set_status("Error — see Log tab")
        QMessageBox.critical(self, "Pipeline Error", msg[:300])

    # ──────────────────────────────────────────
    #  RESULTS
    # ──────────────────────────────────────────
    def _populate_results(self, results):
        dt = results["decision_tree"]["metrics"]
        lr = results["logistic_regression"]["metrics"]

        keys   = ["accuracy", "precision", "recall", "f1_score"]
        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

        self.metrics_table.setRowCount(len(keys))
        for row, (key, lbl) in enumerate(zip(keys, labels)):
            dv = dt.get(key, 0)
            lv = lr.get(key, 0)
            winner = "Logistic Reg ✓" if lv > dv else "Decision Tree ✓"
            win_col = GREEN if lv > dv else YELLOW

            for col, (val, color) in enumerate([
                (lbl,           FG),
                (f"{dv:.4f}",   YELLOW),
                (f"{lv:.4f}",   GREEN),
                (winner,        win_col),
            ]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(color))
                item.setTextAlignment(Qt.AlignCenter)
                self.metrics_table.setItem(row, col, item)

        self.bar_chart.set_data(dt, lr)

        def w(k): return "Logistic Regression" if lr[k] > dt[k] else "Decision Tree"
        txt = "\n".join([
            "INTERPRETATION", "-" * 50,
            f"  Best Accuracy  : {w('accuracy')}",
            f"  Best Precision : {w('precision')}",
            f"  Best Recall    : {w('recall')}",
            f"  Best F1-Score  : {w('f1_score')}",
            "",
            "  Logistic Regression typically generalises better on text data.",
            "  Decision Tree is more interpretable and faster to train.",
        ])
        self.interp_text.setPlainText(txt)

    # ──────────────────────────────────────────
    #  PREDICT
    # ──────────────────────────────────────────
    def _predict_single(self):
        if not self.classifier:
            QMessageBox.warning(self, "Not ready", "Run the pipeline first.")
            return
        hl = self.hl_input.text().strip()
        if not hl:
            QMessageBox.warning(self, "Empty", "Enter a headline.")
            return
        dt = self.classifier.predict(hl, model_type="decision_tree")
        lr = self.classifier.predict(hl, model_type="logistic_regression")
        self.pred_dt.setText(dt)
        self.pred_lr.setText(lr)
        self.pred_agree.setText("✓ Both agree" if dt == lr else "✗ Disagree")
        self._log(f'\nPredict: "{hl}"\n', "cyan")
        self._log(f"  Decision Tree       : {dt}\n", "yellow")
        self._log(f"  Logistic Regression : {lr}\n", "yellow")

    def _run_batch(self):
        if not self.classifier:
            QMessageBox.warning(self, "Not ready", "Run the pipeline first.")
            return
        self.batch_table.setRowCount(0)
        headlines = [
            "Stock market reaches new high",
            "World Cup final draws huge crowd",
            "New AI breakthrough announced",
            "International summit discusses climate",
            "Scientists discover new planet",
            "Government passes new tax legislation",
            "Tech giant reports record profits",
        ]
        self._log("\nBatch prediction:\n", "cyan")
        for h in headlines:
            dt = self.classifier.predict(h, model_type="decision_tree")
            lr = self.classifier.predict(h, model_type="logistic_regression")
            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
            for col, val in enumerate([h, dt, lr]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(FG))
                self.batch_table.setItem(row, col, item)
            self._log(f"  {h[:36]:<36}  DT={dt}  LR={lr}\n", "white")

    # ──────────────────────────────────────────
    #  STEP INDICATORS
    # ──────────────────────────────────────────
    def _update_step(self, idx, state):
        lbl, col = self.step_labels[idx]
        txt = lbl.text()
        if state == "ok":
            txt = txt.replace("○", "✓").replace("●", "✓")
            lbl.setStyleSheet(f"color: {col};")
        elif state == "run":
            txt = txt.replace("○", "●").replace("✓", "●")
            lbl.setStyleSheet(f"color: {YELLOW};")
        lbl.setText(txt)

    def _reset_steps(self):
        for lbl, _ in self.step_labels:
            txt = lbl.text().replace("●", "○").replace("✓", "○")
            lbl.setText(txt)
            lbl.setStyleSheet(f"color: {FG_DIM};")

    # ──────────────────────────────────────────
    def _reset(self):
        self.classifier = None
        self.config_obj = None
        self._reset_steps()
        self._set_status("Ready")
        self.metrics_table.setRowCount(0)
        self.bar_chart.data = None
        self.bar_chart.update()
        self.interp_text.clear()
        self.pred_dt.setText("—")
        self.pred_lr.setText("—")
        self.pred_agree.setText("—")

    def _set_status(self, msg):
        self.status_lbl.setText(msg)

    def _log(self, msg, tag="white"):
        colors = {
            "green": GREEN, "cyan": CYAN, "yellow": YELLOW,
            "red": RED, "dim": FG_DIM, "accent": ACCENT, "white": FG,
        }
        col = colors.get(tag, FG)
        self.log_box.moveCursor(self.log_box.textCursor().End)
        self.log_box.setTextColor(QColor(col))
        self.log_box.insertPlainText(msg)
        self.log_box.moveCursor(self.log_box.textCursor().End)

    def _clear_log(self):
        self.log_box.clear()

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "classifier_log.txt",
            "Text files (*.txt);;All files (*.*)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_box.toPlainText())
            self._log(f"\nLog saved: {path}\n", "green")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = NewsClassifierGUI()
    window.show()
    sys.exit(app.exec_())