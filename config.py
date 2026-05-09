import os
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION MODELS (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class QdrantConfig(BaseModel):
    url: str = ""
    api_key: str = ""
    collection_name: str = "f16_lightrag"
    cosine_better_than_threshold: float = 0.2

class Neo4jConfig(BaseModel):
    uri: str = ""
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

class GroqConfig(BaseModel):
    api_key: str = ""
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 1024
    temperature: float = 0.4
    base_url: str = "https://api.groq.com/openai/v1"
    
    build_model: str = "llama-3.1-8b-instant"
    build_max_tokens: int = 512
    build_temperature: float = 0.2

class EmbeddingConfig(BaseModel):
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768
    max_token_size: int = 2048
    batch_size: int = 16
    max_chars_per_text: int = 2000

class RAGConfig(BaseModel):
    working_dir: str = "./lightrag_qdrant_storage"
    search_modes: List[str] = ["mix", "hybrid", "local", "global", "naive"]
    default_search_mode: str = "local"
    top_k: int = 2
    entity_extract_max_gleaning: int = 1
    enable_llm_cache: bool = True

class AppConfig(BaseModel):
    page_title: str = "F-16 Defence Intelligence Bot"
    allowed_file_types: List[str] = ["pdf", "txt", "md"]
    max_history_turns: int = 3

# ─────────────────────────────────────────────────────────────────────────────
# LOAD LOGIC
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_config_provider():
    """
    Returns a dictionary of validated config objects.
    Checks Streamlit Secrets first, then environment variables (.env).
    """
    # 1. Detect environment
    is_streamlit_cloud = False
    try:
        if hasattr(st, "secrets") and len(st.secrets) > 0:
            is_streamlit_cloud = True
    except:
        pass

    if not is_streamlit_cloud:
        load_dotenv()

    # 2. Helper to get value
    def get_val(key, default=None):
        if is_streamlit_cloud:
            return st.secrets.get(key, os.getenv(key, default))
        return os.getenv(key, default)

    # 3. Instantiate Models
    return {
        "qdrant": QdrantConfig(
            url=get_val("QDRANT_URL", ""),
            api_key=get_val("QDRANT_API_KEY", ""),
            collection_name=get_val("QDRANT_COLLECTION_NAME", "f16_lightrag"),
            cosine_better_than_threshold=float(get_val("QDRANT_COSINE_THRESHOLD", 0.2)),
        ),
        "neo4j": Neo4jConfig(
            uri=get_val("NEO4J_URI", ""),
            username=get_val("NEO4J_USERNAME", "neo4j"),
            password=get_val("NEO4J_PASSWORD", ""),
            database=get_val("NEO4J_DATABASE", "neo4j"),
        ),
        "groq": GroqConfig(
            api_key=get_val("GROQ_API_KEY", ""),
            model=get_val("GROQ_MODEL", "llama-3.3-70b-versatile"),
            build_model=get_val("GROQ_BUILD_MODEL", "llama-3.1-8b-instant"),
        ),
        "embedding": EmbeddingConfig(),
        "rag": RAGConfig(
            working_dir=get_val("RAG_WORKING_DIR", "./lightrag_qdrant_storage"),
        ),
        "app": AppConfig(),
    }

# ─── Global Instances (for easy import) ──────────────────────────────────────
# We call it once here so that 'from config import qdrant_config' still works.
_configs = get_config_provider()

qdrant_config = _configs["qdrant"]
neo4j_config = _configs["neo4j"]
groq_config = _configs["groq"]
embedding_config = _configs["embedding"]
rag_config = _configs["rag"]
app_config = _configs["app"]