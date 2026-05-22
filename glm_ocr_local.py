"""
GLM-OCR Local Runner — CPU Mode
PyQt6 GUI + CLI interface for running zai-org/GLM-OCR locally.

Usage (GUI):
    python glm_ocr_local.py

Usage (CLI):
    python glm_ocr_local.py --image path/to/image.png --mode text
    python glm_ocr_local.py --image invoice.jpg --mode json
    python glm_ocr_local.py --image doc.png --mode latex --output result.txt
"""

import sys
import os
import argparse
import json
import threading
from pathlib import Path

# ─── CLI argument parsing (before Qt import to allow headless use) ──────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="GLM-OCR Local Runner — CPU inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  text    — plain text extraction
  latex   — formulas/math → LaTeX
  table   — tables → Markdown/HTML
  json    — structured info extraction (invoice, form, etc.)
  parse   — full document parsing → Markdown

Examples:
  python glm_ocr_local.py --image invoice.jpg --mode json
  python glm_ocr_local.py --image doc.png --mode latex --output out.txt
  python glm_ocr_local.py  (no args → opens GUI)
        """
    )
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--mode", type=str, default="text",
                        choices=["text", "latex", "table", "json", "parse"],
                        help="OCR mode (default: text)")
    parser.add_argument("--output", type=str, help="Save result to file")
    parser.add_argument("--model", type=str, default="zai-org/GLM-OCR",
                        help="HuggingFace model path (default: zai-org/GLM-OCR)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max output tokens (default: 4096)")
    parser.add_argument("--gui", action="store_true",
                        help="Force open GUI even if --image is given")
    return parser.parse_args()


# ─── Prompt templates per mode ───────────────────────────────────────────────
MODE_PROMPTS = {
    "text":  "Text Recognition:",
    "latex": "Formula Recognition:",
    "table": "Table Recognition:",
    "json":  "Information Extraction:",
    "parse": "Document Parsing:",
}

MODE_LABELS = {
    "text":  "📝 Text Recognition",
    "latex": "∑  LaTeX / Formula",
    "table": "📊 Table Recognition",
    "json":  "🔍 Info Extraction (JSON)",
    "parse": "📄 Full Document Parse",
}


# ─── Core inference function (runs in any context) ───────────────────────────
def run_ocr(image_path: str, mode: str = "text",
            model_id: str = "zai-org/GLM-OCR",
            max_new_tokens: int = 4096,
            progress_callback=None) -> str:
    """
    Load GLM-OCR model and run OCR on image_path.
    Downloads model from HuggingFace on first run (~1.8 GB).
    Runs on CPU — slow but works without GPU.
    """
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch
    except ImportError:
        raise ImportError(
            "Missing dependencies. Run:\n"
            "  pip install transformers>=5.3.0 torch torchvision pillow"
        )

    if progress_callback:
        progress_callback("⏳ Loading model (downloading on first run)…")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,   # float32 for CPU
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    if progress_callback:
        progress_callback("🖼️  Processing image…")

    prompt = MODE_PROMPTS.get(mode, "Text Recognition:")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)

    if progress_callback:
        progress_callback("🧠 Running inference on CPU (please wait)…")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )

    # Clean up special tokens
    output_text = output_text.replace("<|endoftext|>", "").strip()

    if progress_callback:
        progress_callback("✅ Done!")

    return output_text


# ─── CLI mode ────────────────────────────────────────────────────────────────
def run_cli(args):
    if not os.path.exists(args.image):
        print(f"❌ File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Mode   : {MODE_LABELS.get(args.mode, args.mode)}")
    print(f"🖼️  Image  : {args.image}")
    print(f"🤖 Model  : {args.model}")
    print("─" * 50)

    def progress(msg):
        print(msg)

    result = run_ocr(
        image_path=args.image,
        mode=args.mode,
        model_id=args.model,
        max_new_tokens=args.max_tokens,
        progress_callback=progress,
    )

    print("\n" + "═" * 50)
    print("RESULT:")
    print("═" * 50)
    print(result)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"\n💾 Saved to: {args.output}")


# ─── PyQt6 GUI ───────────────────────────────────────────────────────────────
def run_gui(args):
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QPushButton, QLabel, QComboBox, QTextEdit, QFileDialog,
            QProgressBar, QSplitter, QFrame, QScrollArea, QStatusBar,
            QSizePolicy, QMessageBox,
        )
        from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData, QSize
        from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont, QColor, QPalette, QIcon
    except ImportError:
        print("❌ PyQt6 not found. Install with:  pip install PyQt6")
        sys.exit(1)

    # ── Worker thread ─────────────────────────────────────────────────────
    class OcrWorker(QThread):
        progress = pyqtSignal(str)
        finished = pyqtSignal(str)
        error    = pyqtSignal(str)

        def __init__(self, image_path, mode, model_id, max_tokens):
            super().__init__()
            self.image_path = image_path
            self.mode       = mode
            self.model_id   = model_id
            self.max_tokens = max_tokens

        def run(self):
            try:
                result = run_ocr(
                    image_path=self.image_path,
                    mode=self.mode,
                    model_id=self.model_id,
                    max_new_tokens=self.max_tokens,
                    progress_callback=lambda m: self.progress.emit(m),
                )
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))

    # ── Drag-and-drop image label ─────────────────────────────────────────
    class ImageDropLabel(QLabel):
        imageDropped = pyqtSignal(str)

        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(300, 280)
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #3d8ef8;
                    border-radius: 12px;
                    background: #0d1b2a;
                    color: #3d8ef8;
                    font-size: 14px;
                }
            """)
            self.setText("📂 Drop image here\nor click Browse")

        def dragEnterEvent(self, event: QDragEnterEvent):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                self.setStyleSheet("""
                    QLabel {
                        border: 2px solid #00e5ff;
                        border-radius: 12px;
                        background: #0a2540;
                        color: #00e5ff;
                        font-size: 14px;
                    }
                """)

        def dragLeaveEvent(self, event):
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #3d8ef8;
                    border-radius: 12px;
                    background: #0d1b2a;
                    color: #3d8ef8;
                    font-size: 14px;
                }
            """)

        def dropEvent(self, event: QDropEvent):
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".pdf")):
                    self.imageDropped.emit(path)
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #3d8ef8;
                    border-radius: 12px;
                    background: #0d1b2a;
                    color: #3d8ef8;
                    font-size: 14px;
                }
            """)

        def set_image(self, path):
            px = QPixmap(path)
            if not px.isNull():
                px = px.scaled(
                    self.width() - 20, self.height() - 20,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.setPixmap(px)
                self.setText("")
            else:
                self.setText(f"📄 {Path(path).name}")

    # ── Main Window ───────────────────────────────────────────────────────
    class MainWindow(QMainWindow):
        STYLE = """
        QMainWindow, QWidget {
            background-color: #071525;
            color: #c8d8e8;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        QLabel { color: #8ab4d4; }
        QPushButton {
            background: #1a3a5c;
            color: #7ec8f8;
            border: 1px solid #2a5f8a;
            border-radius: 8px;
            padding: 8px 18px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover { background: #2a5a8c; border-color: #3d8ef8; color: #ffffff; }
        QPushButton:pressed { background: #0d2a48; }
        QPushButton:disabled { background: #0d1e30; color: #3a5a7a; border-color: #1a3a5c; }
        QPushButton#runBtn {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1a4a8c, stop:1 #0a2a6c);
            color: #00e5ff;
            border: 1px solid #00a5cf;
            font-size: 14px;
            padding: 10px 24px;
        }
        QPushButton#runBtn:hover {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #2a5aac, stop:1 #1a3a8c);
        }
        QPushButton#copyBtn { background: #0d2a3a; color: #00e5a0; border-color: #00a060; }
        QPushButton#saveBtn { background: #0d2a3a; color: #f0c050; border-color: #c09020; }
        QComboBox {
            background: #0d1b2a; color: #7ec8f8;
            border: 1px solid #2a5f8a; border-radius: 6px;
            padding: 6px 10px; font-size: 13px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background: #0d1b2a; color: #c8d8e8;
            selection-background-color: #1a3a5c;
        }
        QTextEdit {
            background: #05101a; color: #a8d8b8;
            border: 1px solid #1a3a5c; border-radius: 8px;
            font-family: 'Consolas', monospace; font-size: 13px;
            padding: 10px;
        }
        QProgressBar {
            border: 1px solid #1a3a5c; border-radius: 5px;
            background: #0d1b2a; text-align: center; color: #7ec8f8; height: 18px;
        }
        QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 #1a4a8c, stop:1 #00a5cf); border-radius: 4px; }
        QSplitter::handle { background: #1a3a5c; }
        QStatusBar { background: #040e1a; color: #4a7a9a; font-size: 12px; }
        QFrame#card {
            background: #0a1828; border: 1px solid #1a3a5c;
            border-radius: 10px; padding: 8px;
        }
        """

        def __init__(self, initial_image=None, initial_mode="text"):
            super().__init__()
            self.image_path = initial_image
            self.worker     = None

            self.setWindowTitle("GLM-OCR  ·  Local CPU Runner")
            self.setMinimumSize(1000, 680)
            self.setStyleSheet(self.STYLE)

            self._build_ui()
            self.setStatusBar(QStatusBar())
            self.statusBar().showMessage("Ready  •  Model: zai-org/GLM-OCR  •  Device: CPU")

            if initial_image:
                self._load_image(initial_image)
            if initial_mode in MODE_PROMPTS:
                idx = list(MODE_PROMPTS.keys()).index(initial_mode)
                self.mode_combo.setCurrentIndex(idx)

        def _build_ui(self):
            from PyQt6.QtWidgets import (
                QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                QPushButton, QLabel, QComboBox, QTextEdit, QFileDialog,
                QProgressBar, QSplitter, QFrame, QScrollArea, QStatusBar,
                QSizePolicy, QMessageBox,
            )
            from PyQt6.QtCore import Qt

            central = QWidget()
            self.setCentralWidget(central)
            root = QVBoxLayout(central)
            root.setContentsMargins(16, 12, 16, 12)
            root.setSpacing(10)

            # ── Header ──────────────────────────────────────────────────
            header = QLabel(
                "◈  <span style='color:#00e5ff;font-size:20px;font-weight:bold;'>GLM-OCR</span>"
                " <span style='color:#4a7a9a;font-size:13px;'>"
                "0.9B Vision-Language Model  ·  Local CPU</span>"
            )
            header.setTextFormat(Qt.TextFormat.RichText)
            root.addWidget(header)

            # ── Splitter: left (image) | right (output) ─────────────────
            splitter = QSplitter(Qt.Orientation.Horizontal)
            root.addWidget(splitter, stretch=1)

            # Left panel
            left = QFrame()
            left.setObjectName("card")
            left_layout = QVBoxLayout(left)
            left_layout.setSpacing(8)

            lbl_img = QLabel("IMAGE INPUT")
            lbl_img.setStyleSheet("color:#3d8ef8;font-size:11px;font-weight:bold;letter-spacing:2px;")
            left_layout.addWidget(lbl_img)

            self.img_label = ImageDropLabel()
            self.img_label.imageDropped.connect(self._load_image)
            left_layout.addWidget(self.img_label, stretch=1)

            btn_browse = QPushButton("📁  Browse Image…")
            btn_browse.clicked.connect(self._browse_image)
            left_layout.addWidget(btn_browse)

            # Mode selector
            mode_row = QHBoxLayout()
            mode_row.addWidget(QLabel("Mode:"))
            self.mode_combo = QComboBox()
            for key, label in MODE_LABELS.items():
                self.mode_combo.addItem(label, key)
            mode_row.addWidget(self.mode_combo, stretch=1)
            left_layout.addLayout(mode_row)

            # Run button
            self.run_btn = QPushButton("▶  Run OCR")
            self.run_btn.setObjectName("runBtn")
            self.run_btn.clicked.connect(self._run_ocr)
            self.run_btn.setEnabled(False)
            left_layout.addWidget(self.run_btn)

            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 0)   # indeterminate
            self.progress_bar.setVisible(False)
            left_layout.addWidget(self.progress_bar)

            self.progress_lbl = QLabel("")
            self.progress_lbl.setStyleSheet("color:#00a5cf;font-size:12px;")
            self.progress_lbl.setWordWrap(True)
            left_layout.addWidget(self.progress_lbl)

            splitter.addWidget(left)

            # Right panel
            right = QFrame()
            right.setObjectName("card")
            right_layout = QVBoxLayout(right)
            right_layout.setSpacing(8)

            out_header = QHBoxLayout()
            lbl_out = QLabel("OUTPUT")
            lbl_out.setStyleSheet("color:#00e5a0;font-size:11px;font-weight:bold;letter-spacing:2px;")
            out_header.addWidget(lbl_out)
            out_header.addStretch()

            copy_btn = QPushButton("⎘  Copy")
            copy_btn.setObjectName("copyBtn")
            copy_btn.clicked.connect(self._copy_result)
            out_header.addWidget(copy_btn)

            save_btn = QPushButton("💾  Save")
            save_btn.setObjectName("saveBtn")
            save_btn.clicked.connect(self._save_result)
            out_header.addWidget(save_btn)
            right_layout.addLayout(out_header)

            self.output_text = QTextEdit()
            self.output_text.setPlaceholderText(
                "OCR output will appear here…\n\n"
                "Supported modes:\n"
                "  • Text Recognition\n"
                "  • LaTeX / Formula\n"
                "  • Table → Markdown\n"
                "  • Info Extraction → JSON\n"
                "  • Full Document Parse"
            )
            self.output_text.setReadOnly(False)
            right_layout.addWidget(self.output_text, stretch=1)

            splitter.addWidget(right)
            splitter.setSizes([420, 580])

        def _browse_image(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Image",
                str(Path.home()),
                "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.pdf)"
            )
            if path:
                self._load_image(path)

        def _load_image(self, path):
            self.image_path = path
            self.img_label.set_image(path)
            self.run_btn.setEnabled(True)
            self.statusBar().showMessage(f"Loaded: {Path(path).name}")

        def _run_ocr(self):
            if not self.image_path:
                return

            mode = self.mode_combo.currentData()
            self.run_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_lbl.setText("Starting…")
            self.output_text.clear()

            from PyQt6.QtCore import QThread, pyqtSignal

            class OcrWorker(QThread):
                progress = pyqtSignal(str)
                finished = pyqtSignal(str)
                error    = pyqtSignal(str)

                def __init__(self, image_path, mode):
                    super().__init__()
                    self.image_path = image_path
                    self.mode       = mode

                def run(self):
                    try:
                        result = run_ocr(
                            image_path=self.image_path,
                            mode=self.mode,
                            progress_callback=lambda m: self.progress.emit(m),
                        )
                        self.finished.emit(result)
                    except Exception as e:
                        self.error.emit(str(e))

            self.worker = OcrWorker(self.image_path, mode)
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_finished)
            self.worker.error.connect(self._on_error)
            self.worker.start()

        def _on_progress(self, msg):
            self.progress_lbl.setText(msg)
            self.statusBar().showMessage(msg)

        def _on_finished(self, result):
            self.output_text.setPlainText(result)
            self.progress_bar.setVisible(False)
            self.progress_lbl.setText("")
            self.run_btn.setEnabled(True)
            self.statusBar().showMessage(
                f"✅ Done  •  {len(result.split())} words  •  {len(result)} chars"
            )

        def _on_error(self, err):
            self.progress_bar.setVisible(False)
            self.progress_lbl.setText("")
            self.run_btn.setEnabled(True)
            self.output_text.setPlainText(f"❌ Error:\n\n{err}")
            self.statusBar().showMessage("Error — see output panel")

        def _copy_result(self):
            from PyQt6.QtWidgets import QApplication
            text = self.output_text.toPlainText()
            if text:
                QApplication.clipboard().setText(text)
                self.statusBar().showMessage("Copied to clipboard!")

        def _save_result(self):
            from PyQt6.QtWidgets import QFileDialog
            text = self.output_text.toPlainText()
            if not text:
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Result", "ocr_result.txt",
                "Text (*.txt);;Markdown (*.md);;JSON (*.json);;All (*.*)"
            )
            if path:
                Path(path).write_text(text, encoding="utf-8")
                self.statusBar().showMessage(f"Saved → {path}")

    # ── Launch ────────────────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("GLM-OCR Local")

    window = MainWindow(
        initial_image=args.image if args.gui else None,
        initial_mode=args.mode,
    )
    window.show()
    sys.exit(app.exec())


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if args.image and not args.gui:
        # Pure CLI mode
        run_cli(args)
    else:
        # GUI mode (default or --gui flag)
        run_gui(args)
