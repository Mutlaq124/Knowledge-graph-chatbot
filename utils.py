"""
utils.py

LightRAG initialization and shared helpers.

Cloud-ready: uses HuggingFace nomic-embed-text-v1.5 (no Ollama dependency)
and QdrantVectorDBStorage (no local FAISS files).
"""

import logging
import re
import numpy as np
from pathlib import Path
from typing import Optional
from openai import AsyncOpenAI

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.prompt import PROMPTS

from config import qdrant_config, groq_config, rag_config, embedding_config, neo4j_config
from prompt_template import KG_EXTRACTION_PROMPT, DEFENCE_ENTITY_TYPES

PROMPTS["entity_extraction_system_prompt"] = KG_EXTRACTION_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL (CACHED)
# Uses @st.cache_resource when Streamlit is available so the 280 MB model is
# loaded once per app process — not on every query.
# ─────────────────────────────────────────────────────────────────────────────
_embedding_model_instance = None

def get_embedding_model():
    """Load the sentence transformer model once globally."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {embedding_config.model_name}")
        _embedding_model_instance = SentenceTransformer(
            embedding_config.model_name,
            trust_remote_code=True,
            device="cpu",
        )
    return _embedding_model_instance


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
async def hf_embed(texts, *args, **kwargs) -> np.ndarray:
    """Async wrapper around the synchronous SentenceTransformer encode call."""
    model = get_embedding_model()
    # Safety truncation — avoids silent OOM on very long passages
    texts = [t[: embedding_config.max_chars_per_text] for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=embedding_config.batch_size,
        show_progress_bar=False,
    )
    return np.array(embeddings)


def get_embedding_func() -> EmbeddingFunc:
    logger.info(f"Embedding: {embedding_config.model_name} ({embedding_config.embedding_dim}d)")
    return EmbeddingFunc(
        embedding_dim=embedding_config.embedding_dim,
        max_token_size=embedding_config.max_token_size,
        func=hf_embed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM FUNCTION FOR LIGHTRAG KEYWORD EXTRACTION
# LightRAG requires an llm_model_func even at query time to extract 
# keywords from the user's query for graph retrieval.
# ─────────────────────────────────────────────────────────────────────────────
async def async_groq_llm(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    client = AsyncOpenAI(
        api_key=groq_config.api_key,
        base_url=groq_config.base_url,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=groq_config.build_model,
        messages=messages,
        max_tokens=groq_config.build_max_tokens,
        temperature=groq_config.build_temperature,
    )
    return response.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZE LIGHTRAG
# ─────────────────────────────────────────────────────────────────────────────
async def initialize_lightrag() -> Optional[LightRAG]:
    """
    Initialize LightRAG with Qdrant Cloud vector store + Neo4j graph store.

    The working_dir is still used for:
      - LLM response cache (if enable_llm_cache=True)
      - Pipeline status tracking
    No FAISS .index files are created or needed.
    """
    try:
        Path(rag_config.working_dir).mkdir(parents=True, exist_ok=True)

        rag = LightRAG(
            working_dir=rag_config.working_dir,
            embedding_func=get_embedding_func(),
            llm_model_func=async_groq_llm,
            llm_model_name=groq_config.model,
            # Concurrency limits for Groq free tier
            llm_model_max_async=4,
            embedding_func_max_async=4,
            graph_storage="Neo4JStorage",
            vector_storage="QdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                "url": qdrant_config.url,
                "api_key": qdrant_config.api_key,
                "collection_name": qdrant_config.collection_name,
                "cosine_better_than_threshold": qdrant_config.cosine_better_than_threshold
            },
            chunk_token_size=900,
            chunk_overlap_token_size=50,
            entity_extract_max_gleaning=rag_config.entity_extract_max_gleaning,
            enable_llm_cache=rag_config.enable_llm_cache,
            addon_params={"entity_types": DEFENCE_ENTITY_TYPES},
        )

        await rag.initialize_storages()
        await initialize_pipeline_status()

        logger.info("✅ LightRAG initialized (Qdrant Cloud + Neo4j Aura)")
        return rag

    except Exception as e:
        logger.error(f"❌ LightRAG init failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_context_sources(context_str: str) -> list:
    """Extract page/file references from a LightRAG context string."""
    sources = []
    seen = set()

    page_pattern = re.compile(r"===\s*Page\s*(\d+)\s*\|\s*([^=\n]+?)\s*===")
    for match in page_pattern.finditer(context_str):
        page_num = match.group(1)
        filename = match.group(2).strip()
        key = f"{filename} (pg. {page_num})"
        if key not in seen:
            sources.append(key)
            seen.add(key)

    file_pattern = re.compile(r"\b([\w\-]+\.(?:pdf|txt|md))\b", re.IGNORECASE)
    for match in file_pattern.finditer(context_str):
        fname = match.group(1)
        if fname not in seen:
            sources.append(fname)
            seen.add(fname)

    return sources