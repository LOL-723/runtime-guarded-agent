from pydantic import BaseModel, Field


class RagChunk(BaseModel):
    id: str
    index: int
    content: str


class RagUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    chunks: list[RagChunk]


class RagAskRequest(BaseModel):
    question: str = Field(..., description="Question to answer from uploaded documents")
    top_k: int | None = Field(default=None, ge=1, le=20)
    rerank_top_k: int | None = Field(default=None, ge=1, le=10)


class RagSource(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str


class RagAskResponse(BaseModel):
    answer: str
    sources: list[RagSource]
