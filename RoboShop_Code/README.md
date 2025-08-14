# Shelf Restocking VLS — YOLOv8n + Structured JSON LLM (Streamlit)

This app:
1) Runs YOLOv8n on a shelf image to detect products (class, bbox, conf).
2) Summarizes detections.
3) Uses a **structured JSON planner** (Outlines -> Jsonformer -> validated/retry) so the LLM returns *valid JSON*.
4) Converts the valid JSON plan into **robot instructions**.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

### Notes
- First run downloads YOLO weights and `google/flan-t5-small`; then it works offline.
- If RAM is very tight, leave as `small`. You can try `flan-t5-base` by editing `LLM_MODEL_NAME` in `app.py`.
- Outlines/Jsonformer are optional but recommended. If missing or unsupported, the app falls back to a validate+repair+retry loop.
