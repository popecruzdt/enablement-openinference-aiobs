"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    model: str = Field(..., description="Name of the model to use for inference")
    input_text: str = Field(..., description="Input text for inference", min_length=1, max_length=4096)
    max_tokens: int = Field(default=256, description="Maximum number of tokens to generate", ge=1, le=1024)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "text-summarizer-v1",
                    "input_text": "Ansible Automation Platform enables consistent automation across hybrid environments.",
                    "max_tokens": 128,
                }
            ]
        }
    }


class InferenceResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier for tracing")
    model: str = Field(..., description="Model used for inference")
    output_text: str = Field(..., description="Generated output text")
    inference_time_seconds: float = Field(..., description="Time taken for inference in seconds")
    timestamp: str = Field(..., description="Response timestamp in ISO 8601 format")


class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    description: str = Field(..., description="Model description")
    max_tokens: int = Field(..., description="Maximum supported input tokens")
    loaded: bool = Field(..., description="Whether the model is currently loaded and available")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Timestamp in ISO 8601 format")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Application version")
    models_loaded: int = Field(..., description="Number of models currently loaded")


class MetricsResponse(BaseModel):
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Application version")
    total_models: int = Field(..., description="Total number of registered models")
    loaded_models: int = Field(..., description="Number of currently loaded models")
    timestamp: str = Field(..., description="Timestamp in ISO 8601 format")
