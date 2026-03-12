from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    # Increased token limit
    max_new_tokens: int = Field(150, gt=0, le=150)
    temperature: Optional[float] = Field(0.7, gt=0.0, le=1.0)
    top_p: Optional[float] = Field(0.9, gt=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Write one sentence about cloud engineering.",
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            ]
        }
    }

class PredictResponse(BaseModel):
    generated_text: str
    model_version: Optional[str] = None
