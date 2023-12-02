from langchain.pydantic_v1 import BaseModel, Field


class PdfSummary(BaseModel):
    """Summary and meta data of a PDF document."""

    title: str = Field(..., description="Title of the PDF document.")
    classification: str = Field(
        ..., description="Uni class where the PDF document was handed out."
    )
    summary: str = Field(..., description="Summary of the PDF document.")
