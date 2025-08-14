from __future__ import annotations
import json
import re
from typing import List, Optional
from jsonschema import validate as jsonschema_validate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from schemas import PlanItem, Plan, PLAN_JSON_SCHEMA

HAS_OUTLINES = False
HAS_JSONFORMER = False
try:
    import outlines  # type: ignore
    HAS_OUTLINES = True
except Exception:
    pass
try:
    from jsonformer import Jsonformer  # type: ignore
    HAS_JSONFORMER = True
except Exception:
    pass

class StructuredPlanner:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if self.tokenizer.eos_token_id is None and self.tokenizer.sep_token_id is not None:
            self.tokenizer.eos_token_id = self.tokenizer.sep_token_id
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = "cpu"
        try:
            import torch as _t
            self.device = "cuda" if _t.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
        self.hf_model = self.hf_model.to(self.device)
        self._impl_version = "tsv-v2"
        self._last_raw: Optional[str] = None
        self.outlines_model = None

    def _prompt_tsv(self, instruction: str, det_summary: str) -> str:
        return (
            "You are a planner that outputs tabular data ONLY.\n"
            f"Detections: {det_summary}\n"
            f"Goal: {instruction}\n\n"
            "OUTPUT BETWEEN THE MARKERS ONLY.\n"
            "BEGIN\n"
            "A3\tcoke_12oz\t2\n"
            "B1\twater_1L\t4\n"
            "END\n"
            "Rules:\n"
            "- slot_id: non-empty\n"
            "- sku: non-empty\n"
            "- need: integer >= 0\n"
            "Do not output text outside BEGIN/END."
        )

    def _parse_tsv_to_plan(self, text: str) -> List[PlanItem]:
        items: List[PlanItem] = []
        raw = (text or "").replace("\r\n", "\n").strip()
        m = re.search(r"BEGIN\s*(.*?)\s*END", raw, flags=re.S | re.I)
        raw = m.group(1).strip() if m else raw
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        rows: List[List[str]] = []
        if lines and ("," in lines[0] or lines[0].lower().startswith("slot_id")):
            header = [h.strip().lower() for h in re.split(r"[,\t]", lines[0])]
            body = lines[1:] if ("slot_id" in header and "sku" in header and "need" in header) else lines
            for ln in body:
                parts = [p.strip() for p in re.split(r"[,\t]", ln) if p.strip()]
                rows.append(parts)
        else:
            for ln in lines:
                line = ln.strip("` ").strip()
                line = re.sub(r"\b(slot_id|sku|need)\s*=\s*", "", line, flags=re.I)
                parts = [p.strip() for p in line.split("\t") if p.strip()]
                if len(parts) != 3:
                    parts = [p.strip() for p in re.split(r"\s{2,}|\s*\|\s*|,", line) if p.strip()]
                if len(parts) != 3:
                    parts = re.split(r"\s+", line, maxsplit=2)
                rows.append(parts)
        for parts in rows:
            if len(parts) != 3:
                continue
            slot_id, sku, need_s = parts
            if not slot_id or not sku:
                continue
            mnum = re.search(r"-?\d+", need_s)
            if not mnum:
                continue
            need = max(0, int(mnum.group(0)))
            items.append(PlanItem(slot_id=slot_id, sku=sku, need=need))
        obj = [x.model_dump() for x in items]
        jsonschema_validate(instance=obj, schema=PLAN_JSON_SCHEMA)
        return items

    def _generate_tsv_then_build(self, instruction: str, det_summary: str) -> List[PlanItem]:
        prompt = self._prompt_tsv(instruction, det_summary)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(max_new_tokens=220, do_sample=False, num_beams=1,
                          eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
        try:
            import torch
            with torch.no_grad():
                out = self.hf_model.generate(**inputs, **gen_kwargs)
        except Exception:
            out = self.hf_model.generate(**inputs, **gen_kwargs)
        raw = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        self._last_raw = raw
        return self._parse_tsv_to_plan(raw)

    def _prompt_json(self, instruction: str, det_summary: str) -> str:
        return (
            "You are a careful planning assistant.\n"
            f"Detections summary: {det_summary}\n"
            f"Goal: {instruction}\n\n"
            "Return ONLY a JSON array of objects with keys: slot_id, sku, need."
        )

    def _repair_and_validate(self, text: str) -> List[PlanItem]:
        js_start = text.find("[")
        js_end = text.rfind("]") + 1
        if js_start == -1 or js_end <= js_start:
            raise ValueError("No JSON array found in model output")
        cleaned = text[js_start:js_end].replace("'", '"')
        if re.match(r'\[\s*"', cleaned):
            cleaned = re.sub(r'^\[\s*"', '[{"', cleaned)
            cleaned = re.sub(r'"\s*\]$', '"}]', cleaned)
        cleaned = re.sub(r'(\b\w+\b)\s*:', r'"\1":', cleaned)
        obj = json.loads(cleaned)
        jsonschema_validate(instance=obj, schema=PLAN_JSON_SCHEMA)
        return [PlanItem(**x) for x in obj]

    def _retry_freeform(self, prompt: str, max_tries: int = 3, num_beams: int = 4) -> List[PlanItem]:
        try:
            import torch
            no_grad = torch.no_grad()
        except Exception:
            class _Dummy:
                def __enter__(self): ...
                def __exit__(self, *a): ...
            no_grad = _Dummy()
        with no_grad:
            for _ in range(max_tries):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outs = self.hf_model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    num_beams=num_beams, num_return_sequences=min(num_beams, 4),
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id
                )
                for o in outs:
                    raw = self.tokenizer.decode(o, skip_special_tokens=True).strip()
                    try:
                        return self._repair_and_validate(raw)
                    except Exception:
                        continue
        raise ValueError("Could not coerce model output into valid JSON after retries")

    def _target_from_instruction(self, instruction: str, default: int = 6) -> int:
        m = re.search(r"\bto\s+(\d{1,3})\b", (instruction or "").lower())
        return int(m.group(1)) if m else default

    def _baseline_from_summary(self, det_summary: str, instruction: str) -> List[PlanItem]:
        target = self._target_from_instruction(instruction, default=6)
        items: List[PlanItem] = []
        for count, name in re.findall(r"(\d+)\s+([A-Za-z_][A-Za-z0-9_\- ]*)", det_summary or ""):
            c = int(count)
            cls = name.strip().lower().rstrip("(s)").strip()
            need = max(0, target - c) if target else 0
            slot_id = f"A{len(items)+1}"
            sku = cls.replace(" ", "_")
            items.append(PlanItem(slot_id=slot_id, sku=sku, need=need))
        obj = [x.model_dump() for x in items]
        jsonschema_validate(instance=obj, schema=PLAN_JSON_SCHEMA)
        return items

    def generate(self, instruction: str, det_summary: str) -> List[PlanItem]:
        try:
            items = self._generate_tsv_then_build(instruction, det_summary)
            if items:
                return items
        except Exception:
            pass
        if HAS_OUTLINES and self.outlines_model is None:
            try:
                from outlines.models.transformers import Transformers
                self.outlines_model = Transformers(self.model_name)
            except Exception:
                self.outlines_model = None
        if HAS_OUTLINES and self.outlines_model is not None:
            try:
                import outlines
                generator = outlines.generate.json(self.outlines_model, Plan)
                result = generator(self._prompt_json(instruction, det_summary))
                jsonschema_validate(instance=[x.model_dump() for x in result], schema=PLAN_JSON_SCHEMA)
                return result
            except Exception:
                pass
        if HAS_JSONFORMER:
            try:
                jf = Jsonformer(
                    model=self.hf_model,
                    tokenizer=self.tokenizer,
                    json_schema=PLAN_JSON_SCHEMA,
                    prompt=self._prompt_json(instruction, det_summary),
                    max_array_length=16,
                )
                obj = jf()
                jsonschema_validate(instance=obj, schema=PLAN_JSON_SCHEMA)
                return [PlanItem(**x) for x in obj]
            except Exception:
                pass
        try:
            items = self._retry_freeform(self._prompt_json(instruction, det_summary))
            if items:
                return items
        except Exception:
            pass
        return self._baseline_from_summary(det_summary, instruction)

__all__ = ["StructuredPlanner"]
