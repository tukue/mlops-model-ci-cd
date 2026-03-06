from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(20, gt=0, le=100)
    temperature: Optional[float] = Field(0.7, gt=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Write one sentence about cloud engineering.",
                    "max_new_tokens": 50,
                    "temperature": 0.8,
                }
            ]
        }
    }

class PredictResponse(BaseModel):
    generated_text: str
    model_version: Optional[str] = None
