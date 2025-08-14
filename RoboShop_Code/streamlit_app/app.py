import streamlit as st
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from ultralytics import YOLO
from llm_structured import StructuredPlanner
from schemas import PlanItem


YOLO_MODEL_PATH = "yolov8n.pt"
LLM_MODEL_NAME = "google/flan-t5-small"

@st.cache_resource
def load_yolo(model_path: str = YOLO_MODEL_PATH):
    return YOLO(model_path)

@st.cache_resource
def load_planner(model_name: str = LLM_MODEL_NAME, impl_version: str = "tsv-v1"):
    return StructuredPlanner(model_name=model_name)

yolo_model = load_yolo()
planner = load_planner()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def get_saved_annotated_path() -> str:
    """Return the most recent annotated image path saved by Ultralytics."""
    save_dir = Path("outputs/annotated/results")
    if not save_dir.exists():
        return ""
    imgs = sorted(
        list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(imgs[0]) if imgs else ""

def det_summary_text(dets: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for d in dets:
        k = d["class"]
        counts[k] = counts.get(k, 0) + 1
    return ", ".join(f"{v} {k}(s)" for k, v in counts.items()) if counts else "none"

def to_robot_actions(plan_items: List[PlanItem]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for p in plan_items:
        if p.need > 0:
            sid = p.slot_id
            sku = p.sku
            need = p.need
            steps += [
                {"action": "navigate", "target": sid},
                {"action": "pick", "item": sku, "qty": need},
                {"action": "place", "item": sku, "location": sid},
                {"action": "verify", "slot": sid, "metric": "facing_count"},
            ]
    return steps

def run_yolo_on_image(image_bytes: bytes, conf: float = 0.25):
    """Run detection on uploaded bytes, save annotated image, and return (dets, ann_path)."""
    ensure_dir(Path("outputs/annotated/results"))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    results = yolo_model.predict(
        source=tmp_path,
        save=True,
        project="outputs/annotated",
        name="results",
        exist_ok=True,
        conf=conf,
        verbose=False,
    )

    dets = []
    try:
        r = results[0]
        names = r.names if hasattr(r, "names") else (getattr(yolo_model, "names", {}) or {})
        names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else {}
        for b in r.boxes:
            xyxy = [float(v) for v in b.xyxy[0].tolist()]
            cls_i = int(b.cls)
            dets.append({"class": names.get(cls_i, str(cls_i)), "conf": float(b.conf), "bbox": xyxy})
    except Exception:
        pass

    ann_path = get_saved_annotated_path()
    return dets, ann_path

st.set_page_config(page_title="RoboShop", layout="centered")
st.title("RoboShop")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
    st.caption("Lower = more detections (and more false positives).")

uploaded_img = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png"])
instruction = st.text_input("Restocking Instruction", value="Refill all detected items to 6")
run_btn = st.button("Submit")

if run_btn:
    if not uploaded_img or not instruction.strip():
        st.warning("Please upload an image and enter an instruction.")
        st.stop()

    with st.spinner("Running detection..."):
        img_bytes = uploaded_img.read()
        dets, ann_path = run_yolo_on_image(img_bytes, conf=conf)

    st.subheader("Detections")
    st.write(det_summary_text(dets) or "none")

    if ann_path:
        st.image(ann_path, caption="Detected Products", use_container_width=True)

    st.subheader("Planning")
    summary = det_summary_text(dets)
    with st.spinner("Generating plan..."):
        plan_items: List[PlanItem] = planner.generate(instruction, summary)

    if not plan_items:
        st.warning("No actionable items were generated. Try a clearer instruction!!")
        with st.expander("Debug: model raw output", expanded=False):
            st.code(getattr(planner, "_last_raw", "") or "(empty)", language="text")
        st.stop()

    st.subheader("Restock Plan")
    st.json([p.model_dump() for p in plan_items])

    st.subheader("Robot Instructions")
    st.json(to_robot_actions(plan_items))
