from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DEEPSEEK_API_KEY:str
    DEEPSEEK_BASE_URL:str
    LLM_MODEL:str
    LLM_TIMEOUT:float
    RAG_STORAGE_DIR: str = "storage/rag"
    RAG_CHROMA_DIR: str = "storage/chroma_db"
    RAG_COLLECTION_NAME: str = "default"
    RAG_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RAG_RERANK_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 80
    RAG_RETRIEVE_TOP_K: int = 5
    RAG_RERANK_TOP_K: int = 3
    RAG_MAX_UPLOAD_MB: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
