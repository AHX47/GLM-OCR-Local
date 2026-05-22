# GLM‑OCR Local Runner

Run the powerful **zai-org/GLM-OCR** (0.9B vision‑language model) completely **offline** on your **CPU** – no GPU required. But better  using **GPU** 
Includes a modern PyQt6 GUI and a command‑line interface.  

[→ English](#english) | [→ العربية](#arabic)

---

## English

### Features
- ✅ **100% local** – works without internet after first download
- ✅ **CPU‑only** – uses PyTorch with float32, runs on any machine
- ✅ **Multiple OCR modes**:
  - `text` – plain text extraction  
  - `latex` – mathematical formulas → LaTeX  
  - `table` – tables → Markdown/HTML  
  - `json` – structured information extraction (invoices, forms)  
  - `parse` – full document parsing → Markdown  
- ✅ **Modern GUI** (PyQt6) with drag‑and‑drop image support
- ✅ **CLI** for scripting / headless use
- ✅ **First‑run auto‑download** – model (~1.8 GB) is fetched from HuggingFace Hub

### Requirements
- Python 3.10 or higher
- ~4 GB free RAM (for model + processing)
- ~2 GB disk space (for model cache)

### Installation

#### 1. Clone or download the project
```bash
git clone https://github.com/yourusername/glm-ocr-local.git
cd glm-ocr-local
2. Run the setup script
Linux / macOS – chmod +x run_linux.sh && ./run_linux.sh

Windows – double‑click run_windows.bat

The script creates a virtual environment, installs dependencies (PyTorch, transformers, PyQt6, …) and launches the GUI.

3. Manual installation (optional)
bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python glm_ocr_local.py
Usage
GUI mode (default)
bash
python glm_ocr_local.py
Drag & drop an image (PNG, JPG, PDF, …) or click Browse

Choose the OCR mode from the dropdown

Click Run OCR – the result appears on the right

Copy to clipboard or save as .txt / .md / .json

CLI mode (for automation)
bash
# Basic text extraction
python glm_ocr_local.py --image document.png --mode text

# Extract table as Markdown and save to file
python glm_ocr_local.py --image invoice.jpg --mode table --output result.md

# Full document parse
python glm_ocr_local.py --image report.pdf --mode parse

# Get JSON output
python glm_ocr_local.py --image form.png --mode json
CLI arguments:

--image – path to image/PDF

--mode – one of text, latex, table, json, parse

--output – save result to file

--model – change HuggingFace model (default: zai-org/GLM-OCR)

--max-tokens – limit output length (default 4096)

--gui – force GUI even if --image is given

How it works
Loads the GLM‑OCR model from HuggingFace (first run downloads ~1.8 GB to ~/.cache/huggingface/)

Processor applies the correct system prompt according to the chosen mode

Model runs on CPU – slower than GPU, but fully offline and private

Output is cleaned from special tokens and displayed / saved

Troubleshooting
Issue	Solution
ModuleNotFoundError: No module named 'torch'	Run pip install -r requirements.txt inside the virtual environment
Out of memory (RAM)	Close other applications. The model needs ~3‑4 GB RAM on CPU.
GUI doesn't start	Make sure PyQt6 is installed: pip install PyQt6
Model download fails	Check your internet connection. You can manually download from HuggingFace and place in ~/.cache/huggingface/hub/
Very slow inference	CPU inference is expected to be slow (10‑60 seconds per image). For faster results use a GPU or a smaller model.
Credits
Model: zai-org/GLM-OCR – 0.9B vision‑language model

Built with HuggingFace Transformers and PyTorch

GUI: PyQt6

License
MIT

العربية
مشغل GLM-OCR المحلي
شغّل نموذج zai-org/GLM-OCR (0.9 مليار معلمة) محلياً بالكامل على وحدة المعالجة المركزية (CPU) – لا حاجة لبطاقة رسوميات.
يتضمن واجهة رسومية حديثة (PyQt6) وواجهة سطر أوامر.

الميزات
✅ يعمل محلياً – لا يحتاج إنترنت بعد التحميل الأول

✅ يعمل على CPU فقط – يستخدم PyTorch مع float32، يعمل على أي حاسوب

✅ وضعيات OCR متعددة:

text – استخراج النص العادي

latex – تحويل المعادلات الرياضية إلى لاتيك

table – تحويل الجداول إلى Markdown أو HTML

json – استخراج معلومات منظمة (فواتير، نماذج)

parse – تحليل كامل للوثيقة إلى Markdown

✅ واجهة رسومية حديثة (PyQt6) مع دعم السحب والإفلات للصور

✅ واجهة سطر أوامر للأتمتة والبرمجة النصية

✅ تحميل تلقائي في أول تشغيل – النموذج (حوالي 1.8 جيجابايت) يُجلب من HuggingFace Hub

المتطلبات
Python 3.10 أو أحدث

ذاكرة وصول عشوائي (RAM) حرة حوالي 4 جيجابايت

مساحة تخزين حوالي 2 جيجابايت (لذاكرة النموذج المخبأة)

التثبيت
1. استنساخ أو تحميل المشروع
bash
git clone https://github.com/yourusername/glm-ocr-local.git
cd glm-ocr-local
2. تشغيل سكربت الإعداد
لينكس / macOS – chmod +x run_linux.sh && ./run_linux.sh

ويندوز – انقر نقراً مزدوجاً على run_windows.bat

السكربت ينشئ بيئة افتراضية، يثبت المكتبات المطلوبة (PyTorch، transformers، PyQt6، …) ثم يفتح الواجهة الرسومية.

3. التثبيت اليدوي (اختياري)
bash
python -m venv .venv
source .venv/bin/activate      # أو .venv\Scripts\activate على ويندوز
pip install -r requirements.txt
python glm_ocr_local.py
الاستخدام
الواجهة الرسومية (الافتراضي)
bash
python glm_ocr_local.py
اسحب وأفلت صورة (PNG، JPG، PDF، …) أو اضغط Browse

اختر وضعية OCR من القائمة المنسدلة

اضغط Run OCR – تظهر النتيجة على اليمين

يمكن نسخ النتيجة أو حفظها كملف .txt / .md / .json

سطر الأوامر (للأتمتة)
bash
# استخراج نص عادي
python glm_ocr_local.py --image document.png --mode text

# استخراج جدول وحفظه كـ Markdown
python glm_ocr_local.py --image invoice.jpg --mode table --output result.md

# تحليل كامل للوثيقة
python glm_ocr_local.py --image report.pdf --mode parse

# الحصول على مخرجات JSON
python glm_ocr_local.py --image form.png --mode json
معاملات سطر الأوامر:

--image – مسار الصورة أو ملف PDF

--mode – أحد القيم text، latex، table، json، parse

--output – حفظ النتيجة إلى ملف

--model – تغيير نموذج HuggingFace (الافتراضي zai-org/GLM-OCR)

--max-tokens – تحديد أقصى طول للمخرجات (الافتراضي 4096)

--gui – فتح الواجهة الرسومية حتى مع وجود --image

كيف يعمل
يقوم بتحميل نموذج GLM-OCR من HuggingFace (أول تشغيل: تحميل ~1.8 جيجابايت إلى ~/.cache/huggingface/)

المعالج (Processor) يطالب النموذج بالتعليمات حسب الوضعية المختارة

النموذج يعمل على CPU – أبطأ من GPU لكنه خاص وآمن تماماً

يتم تنظيف المخرجات من الرموز الخاصة وعرضها أو حفظها

حل المشاكل الشائعة
المشكلة	الحل
خطأ No module named 'torch'	نفذ pip install -r requirements.txt داخل البيئة الافتراضية
نفاد الذاكرة (RAM)	أغلق التطبيقات الأخرى. النموذج يحتاج ~3-4 جيجابايت على CPU
الواجهة الرسومية لا تظهر	تأكد من تثبيت PyQt6: pip install PyQt6
فشل تحميل النموذج	تحقق من اتصال الإنترنت. يمكنك تحميل النموذج يدوياً من HuggingFace ووضعه في ~/.cache/huggingface/hub/
الأداء بطيء جداً	تشغيل النموذج على CPU بطيء متوقع (10-60 ثانية لكل صورة). للحصول على سرعة أفضل استخدم GPU أو نموذجاً أصغر
الشكر والإسناد
النموذج: zai-org/GLM-OCR – نموذج رؤية‑لغة بحجم 0.9 مليار معلمة

تم البناء باستخدام HuggingFace Transformers و PyTorch

الواجهة الرسومية: PyQt6

الرخصة
MIT
