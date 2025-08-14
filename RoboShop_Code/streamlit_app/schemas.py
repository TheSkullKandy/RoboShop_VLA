from pydantic import BaseModel, Field, conint
from typing import List

class PlanItem(BaseModel):
    slot_id: str = Field(..., min_length=1)
    sku: str = Field(..., min_length=1)
    need: conint(ge=0)

Plan = List[PlanItem]

PLAN_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["slot_id", "sku", "need"],
        "properties": {
            "slot_id": {"type": "string", "minLength": 1},
            "sku": {"type": "string", "minLength": 1},
            "need": {"type": "integer", "minimum": 0}
        },
        "additionalProperties": False
    }
}
