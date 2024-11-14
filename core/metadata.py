from typing import Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    """Detailed metadata for a document to improve its searchability and contextual understanding."""

    subject: Optional[str] = Field(
        default=None, description="Title of the document, providing a clear, concise representation of its content. Maximum 3 sentences"
    )
    keywords: Optional[str] = Field(
        default=None, description="Keywords that characterize the document; useful for search and categorization. Maximum to be 30 keywords."
    )
