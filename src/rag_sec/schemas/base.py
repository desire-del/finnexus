from pydantic import BaseModel, ConfigDict


class FinNexusSchema(BaseModel):
    """Base schema for FinNexus models. All other schemas should inherit from this class."""

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields not defined in the schema
        from_attributes=True,  # Allow initialization from attributes
        str_strip_whitespace=True
    )