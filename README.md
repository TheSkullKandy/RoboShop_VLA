# Shelf Restocking AI — YOLO + Structured LLM

A computer vision + language model pipeline for automated shelf restocking planning.

This app uses:
- **YOLOv8** (Ultralytics) for product detection.
- **Flan-T5** (or other Seq2Seq LLM) for generating a structured, JSON-valid restocking plan.
- A **schema validator** to guarantee valid output for downstream robotics or inventory systems.

## Features
- Object detection on shelf images (bottles, boxes, etc.).
- Generates a valid JSON plan with:
  - `slot_id` (location)
  - `sku` (product type)
  - `need` (quantity to restock)
- Converts plans to robot-friendly action steps.
- Works on CPU or GPU.
- Failsafe mode: falls back to a simple rule-based planner if LLM parsing fails.

## Project Structure
```
restock_vls_app_structured/
├── streamlit_app/
│   ├── app.py              # Streamlit UI
│   ├── llm_structured.py   # StructuredPlanner class
│   ├── schemas.py          # JSON schema + data models
├── outputs/                # Annotated YOLO results
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone and create virtual environment
```bash
git clone <your-repo-url>
cd restock_vls_app_structured
python -m venv .venv
source .venv/bin/activate   # On Mac/Linux
# .venv\Scripts\activate  # On Windows
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. (Optional) Install with GPU support
If you have CUDA installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Running the App
From the project root:
```bash
streamlit run streamlit_app/app.py
```
Streamlit will print a local URL (e.g., `http://localhost:8501`) — open it in your browser.

## Usage
1. Upload a shelf image (JPG/PNG).
2. Enter a restocking instruction (e.g., `Refill all detected items to 6`).
3. Click "Submit".
4. The app will:
   - Detect products using YOLO.
   - Summarize counts.
   - Generate a schema-valid restock plan.
   - Display robot action steps.

## How It Works
1. **YOLOv8 Detection**
   - Runs object detection and saves annotated results.
2. **Structured LLM Planning**
   - First tries a TSV-first parsing approach for robustness.
   - Falls back to freeform JSON parsing if needed.
   - Validates output with JSON Schema.
3. **Robot Action Generation**
   - Converts each plan item into `navigate → pick → place → verify` steps.

## Requirements
Key dependencies:
- `ultralytics` — YOLOv8 object detection.
- `transformers` — Hugging Face models (Flan-T5 by default).
- `jsonschema` — Output validation.
- `streamlit` — UI.

See `requirements.txt` for full list.

## License
This project is released under the MIT License.

## Contributing
Pull requests are welcome.  
Please make sure changes are tested and code style is followed.

## Troubleshooting
- **ModuleNotFoundError: streamlit_app**  
  Make sure you run from the project root:
  ```bash
  streamlit run streamlit_app/app.py
  ```
- **YOLO model not found**  
  First run will auto-download `yolov8n.pt`.
- **LLM too slow on CPU**  
  Use a smaller model (`google/flan-t5-small`) or switch to GPU.
