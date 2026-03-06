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
                    "prompt": "Define Minimum viable product about ecommerce saas.",
                    "max_new_tokens": 50,
                    "temperature": 0.5,
                }
            ]
        }
    }

class PredictResponse(BaseModel):
    request_id: str
    generated_text: str
    model_version: Optional[str] = None

class FeedbackRequest(BaseModel):
    request_id: str
    score: int = Field(..., ge=1, le=5, description="User feedback score (1-5)")
    comment: Optional[str] = None
