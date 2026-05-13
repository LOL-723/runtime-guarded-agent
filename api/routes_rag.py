from fastapi import APIRouter, HTTPException, UploadFile, status

from llm.rag_service import RagError, rag_service
from schemas.rag import RagAskRequest, RagAskResponse, RagUploadResponse


router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/upload", response_model=RagUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile):
    try:
        return await rag_service.upload(file)
    except RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG upload failed: {exc}") from exc


@router.post("/ask", response_model=RagAskResponse)
def ask_document(req: RagAskRequest):
    try:
        return rag_service.ask(
            question=req.question,
            top_k=req.top_k,
            rerank_top_k=req.rerank_top_k,
        )
    except RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG ask failed: {exc}") from exc
