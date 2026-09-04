"""Declare image-neighbour lookup without registering a Phase 2 implementation."""

from pydantic import BaseModel, ConfigDict, Field

IMAGE_VECTOR_NAME = "image"


class FindSimilarObjectsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    object_id: int = Field(
        gt=0, description="Met Object ID whose published image vector is the query"
    )
    k: int = Field(default=5, ge=1, le=50, description="Maximum number of other objects to return")


FIND_SIMILAR_OBJECTS_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "find_similar_objects",
        "description": (
            "Find visually similar collection objects using the named image vector and cosine "
            "similarity. Use the source object's published vector, exclude the source object, "
            "and return at most k neighbours. If the object has no published image vector, "
            "report image_unavailable; do not substitute text similarity."
        ),
        "parameters": FindSimilarObjectsArguments.model_json_schema(),
    },
}
