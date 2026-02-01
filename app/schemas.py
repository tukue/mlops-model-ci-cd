from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=4, max_length=4)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [5.1, 3.5, 1.4, 0.2]
                }
            ]
        }
    }

class PredictResponse(BaseModel):
    prediction: int
