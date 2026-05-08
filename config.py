import os
from dataclasses import dataclass, field
from typing import List, Dict, Literal
from dotenv import load_dotenv

load_dotenv()


@dataclass
class QdrantConfig:
    """Qdrant Cloud vector store configuration."""
    url: str = os.getenv("QDRANT_URL", "")
    api_key: str = os.getenv("QDRANT_API_KEY", "")
    # LightRAG auto-creates the collection on first run — no manual Qdrant UI setup needed.
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "f16_lightrag")
    # Cosine similarity threshold; lower = more results, less precise.
    # 0.2 is a safe default for nomic-embed-text-v1.5 on defence documents.
    cosine_better_than_threshold: float = float(
        os.getenv("QDRANT_COSINE_THRESHOLD", "0.2")
    )


@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "")
    username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass
class GroqConfig:
    api_key: str = os.getenv("GROQ_API_KEY", "")
    # llama-3.3-70b-versatile: 6,000 TPM / 30 RPM / 14,400 TPD on free tier
    model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    max_tokens: int = 1024   # ~1K output keeps TPM budget healthy at 6K TPM
    temperature: float = 0.4
    base_url: str = "https://api.groq.com/openai/v1"

    # Used during index build (entity extraction) — keep low to avoid rate limits
    build_model: str = os.getenv("GROQ_BUILD_MODEL", "llama-3.1-8b-instant")
    build_max_tokens: int = 512
    build_temperature: float = 0.2


@dataclass
class EmbeddingConfig:
    """HuggingFace embedding — works locally and on Streamlit Cloud (no Ollama required)."""
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768          # fixed for nomic-embed-text-v1.5
    max_token_size: int = 2048
    batch_size: int = 16
    # Truncate texts before encoding to avoid silent OOM on long chunks
    max_chars_per_text: int = 2000


@dataclass
class RAGConfig:
    working_dir: str = os.getenv("RAG_WORKING_DIR", "./lightrag_qdrant_storage")
    search_modes: List[Literal["mix", "hybrid", "local", "global", "naive"]] = field(
        default_factory=lambda: ["mix", "hybrid", "local", "global", "naive"]
    )
    default_search_mode: str = "local"
    top_k: int = 3
    # How many refinement passes during KG entity extraction (1 = fast, 2 = thorough)
    entity_extract_max_gleaning: int = 1
    enable_llm_cache: bool = True


@dataclass
class AppConfig:
    page_title: str = "F-16 Defence Intelligence Bot"
    allowed_file_types: List[str] = field(default_factory=lambda: ["pdf", "txt", "md"])
    max_history_turns: int = 3


# ─── Singletons ───────────────────────────────────────────────────────────────
qdrant_config = QdrantConfig()
neo4j_config = Neo4jConfig()
groq_config = GroqConfig()
embedding_config = EmbeddingConfig()
rag_config = RAGConfig()
app_config = AppConfig()