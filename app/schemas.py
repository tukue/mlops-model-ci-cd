from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(150, gt=0, le=150)
    # Added ge=0.1 to prevent instability
    temperature: Optional[float] = Field(0.7, ge=0.1, le=1.0)
    # Added ge=0.1 to prevent instability
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0)
    # Exposed top_k to make it configurable
    top_k: Optional[int] = Field(50, gt=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Write one sentence about cloud engineering.",
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                }
            ]
        }
    }

class PredictResponse(BaseModel):
    generated_text: str
    model_version: Optional[str] = None
