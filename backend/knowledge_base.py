import os
import re
import json
import uuid
from typing import List, Optional, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI, AsyncOpenAI
from loguru import logger
import tiktoken

# Default embedding model; override with OPENAI_EMBEDDING_MODEL
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Configure loguru for knowledge base operations
logger.add(
    "logs/knowledge_base.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
)


# =============================================================================
# Custom Exceptions
# =============================================================================

class KnowledgeBaseError(Exception):
    """Base exception for knowledge base operations."""
    pass


class EmbeddingError(KnowledgeBaseError):
    """Raised when embedding generation fails."""
    pass


class SearchError(KnowledgeBaseError):
    """Raised when search operation fails."""
    pass


class IngestionError(KnowledgeBaseError):
    """Raised when document ingestion fails."""
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def _format_vector(vec: List[float]) -> str:
    """Convert a list of floats to pgvector literal format."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def redact_text(text: str) -> str:
    """Mask NPIs, Tax IDs, phone numbers to avoid storing raw identifiers."""
    patterns = [
        r"\b\d{10}\b",                # NPI / 10-digit blocks
        r"\b\d{3}-\d{2}-\d{4}\b",     # SSN-like
        r"\b\d{2}-\d{7}\b",           # Tax ID format
        r"\b\d{3}-\d{3}-\d{4}\b",     # Phone
        r"\b\d{3}\d{3}\d{4}\b",       # Phone digits only
    ]
    redacted = text
    for pattern in patterns:
        redacted = re.sub(pattern, "***REDACTED***", redacted)
    return redacted


# =============================================================================
# Text Chunker
# =============================================================================

class TextChunker:
    """Chunk text into overlapping segments for embedding."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        model: str = "text-embedding-3-small"
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks based on token count."""
        tokens = self.encoder.encode(text)

        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start forward by (chunk_size - overlap)
            start += self.chunk_size - self.overlap

            # Prevent infinite loop on very small remaining text
            if start >= len(tokens):
                break

        return chunks

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))


# =============================================================================
# Knowledge Base Class
# =============================================================================

class KnowledgeBase:
    """pgvector-backed knowledge store with chunking and async support."""

    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("SUPABASE_HOST"),
                database="postgres",
                user=os.getenv("SUPABASE_USER", "postgres"),
                password=os.getenv("SUPABASE_PASSWORD"),
                port=5432,
            )
            self.openai_sync = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.openai_async = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.chunker = TextChunker(chunk_size=500, overlap=100, model=EMBED_MODEL)
            logger.info("KnowledgeBase initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeBase: {e}")
            raise KnowledgeBaseError(f"Initialization failed: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
            logger.debug("KnowledgeBase connection closed")

    # -------------------------------------------------------------------------
    # Synchronous Embedding Methods
    # -------------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Synchronous embedding (for backward compatibility)."""
        try:
            resp = self.openai_sync.embeddings.create(
                model=EMBED_MODEL,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"Sync embedding failed: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding for multiple texts."""
        if not texts:
            return []

        try:
            batch_size = 100  # OpenAI supports up to 2048, but 100 is safer
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = self.openai_sync.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Batch embedded {len(batch)} texts")

            return all_embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(f"Batch embedding failed: {e}")

    # -------------------------------------------------------------------------
    # Asynchronous Embedding Methods
    # -------------------------------------------------------------------------

    async def embed_async(self, text: str) -> List[float]:
        """Asynchronous embedding for non-blocking operations."""
        try:
            resp = await self.openai_async.embeddings.create(
                model=EMBED_MODEL,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"Async embedding failed: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous batch embedding for multiple texts (cost-efficient)."""
        if not texts:
            return []

        try:
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = await self.openai_async.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Async batch embedded {len(batch)} texts")

            return all_embeddings
        except Exception as e:
            logger.error(f"Async batch embedding failed: {e}")
            raise EmbeddingError(f"Batch embedding failed: {e}")

    # -------------------------------------------------------------------------
    # Ingestion Methods
    # -------------------------------------------------------------------------

    def upsert_entry(
        self,
        insurance_name: str,
        provider_name: Optional[str],
        call_id: Optional[str],
        request_id: Optional[str],
        summary: str,
        qa_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Ingest a call summary with automatic chunking.
        For large texts, creates multiple rows with chunk_index and parent_id.
        """
        try:
            combined = redact_text(f"{summary}\n{qa_text}".strip())

            # Chunk the text
            chunks = self.chunker.chunk_text(combined)
            logger.info(f"Ingesting {len(chunks)} chunk(s) for call {call_id}")

            if len(chunks) == 1:
                # Single chunk - simple insert (backward compatible)
                embedding = self.embed(combined)
                embedding_literal = _format_vector(embedding)

                with self.conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO call_knowledge
                        (insurance_name, provider_name, call_id, request_id, summary, qa_text, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                        """,
                        (
                            insurance_name,
                            provider_name,
                            call_id,
                            request_id,
                            combined,
                            qa_text,
                            embedding_literal,
                            json.dumps(metadata or {}),
                        ),
                    )
                    self.conn.commit()
            else:
                # Multiple chunks - batch embed and insert with parent_id
                embeddings = self.embed_batch(chunks)
                parent_id = None

                with self.conn.cursor() as cur:
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        embedding_literal = _format_vector(embedding)
                        entry_id = str(uuid.uuid4())

                        if i == 0:
                            parent_id = entry_id

                        cur.execute(
                            """
                            INSERT INTO call_knowledge
                            (id, insurance_name, provider_name, call_id, request_id,
                             summary, qa_text, embedding, metadata, chunk_index, parent_id, total_chunks)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s)
                            """,
                            (
                                entry_id,
                                insurance_name,
                                provider_name,
                                call_id,
                                request_id,
                                chunk if i == 0 else f"[Chunk {i+1}/{len(chunks)}] {chunk[:100]}...",
                                qa_text if i == 0 else "",
                                embedding_literal,
                                json.dumps(metadata or {}),
                                i,
                                parent_id,
                                len(chunks),
                            ),
                        )

                    self.conn.commit()
                    logger.info(f"Successfully ingested {len(chunks)} chunks for call {call_id}")

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Ingestion failed for call {call_id}: {e}")
            raise IngestionError(f"Failed to ingest entry: {e}")

    async def upsert_entry_async(
        self,
        insurance_name: str,
        provider_name: Optional[str],
        call_id: Optional[str],
        request_id: Optional[str],
        summary: str,
        qa_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Async version of upsert_entry with chunking and batch embedding.
        """
        try:
            combined = redact_text(f"{summary}\n{qa_text}".strip())

            # Chunk the text
            chunks = self.chunker.chunk_text(combined)
            logger.info(f"Async ingesting {len(chunks)} chunk(s) for call {call_id}")

            # Batch embed all chunks asynchronously
            embeddings = await self.embed_batch_async(chunks)

            parent_id = None

            with self.conn.cursor() as cur:
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    embedding_literal = _format_vector(embedding)
                    entry_id = str(uuid.uuid4())

                    if i == 0:
                        parent_id = entry_id

                    cur.execute(
                        """
                        INSERT INTO call_knowledge
                        (id, insurance_name, provider_name, call_id, request_id,
                         summary, qa_text, embedding, metadata, chunk_index, parent_id, total_chunks)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s)
                        """,
                        (
                            entry_id,
                            insurance_name,
                            provider_name,
                            call_id,
                            request_id,
                            chunk if i == 0 else f"[Chunk {i+1}/{len(chunks)}] {chunk[:100]}...",
                            qa_text if i == 0 else "",
                            embedding_literal,
                            json.dumps(metadata or {}),
                            i,
                            parent_id,
                            len(chunks),
                        ),
                    )

                self.conn.commit()
                logger.info(f"Async successfully ingested {len(chunks)} chunks for call {call_id}")

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Async ingestion failed for call {call_id}: {e}")
            raise IngestionError(f"Failed to ingest entry: {e}")

    # -------------------------------------------------------------------------
    # Retrieval Methods
    # -------------------------------------------------------------------------

    def search(
        self,
        insurance_name: str,
        provider_name: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant knowledge entries.
        Uses vector similarity when query is provided, otherwise returns recent entries.
        """
        try:
            embedding_literal = None
            if query:
                emb = self.embed(query)
                embedding_literal = _format_vector(emb)

            filters = ["insurance_name ILIKE %s"]
            params: List[Any] = [f"%{insurance_name}%"]
            if provider_name:
                filters.append("provider_name ILIKE %s")
                params.append(f"%{provider_name}%")

            # Only get parent chunks (chunk_index = 0 or NULL for backward compat)
            filters.append("(chunk_index = 0 OR chunk_index IS NULL)")

            where_clause = " AND ".join(filters)
            order_clause = "ORDER BY created_at DESC"
            if embedding_literal:
                order_clause = "ORDER BY embedding <-> %s::vector"
                params.append(embedding_literal)

            params.append(limit)

            sql = f"""
                SELECT id, insurance_name, provider_name, call_id, request_id,
                       summary, qa_text, metadata, created_at
                FROM call_knowledge
                WHERE {where_clause}
                {order_clause}
                LIMIT %s
            """

            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                logger.debug(f"Search returned {len(rows)} results for insurance={insurance_name}")
                return [dict(r) for r in rows]

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search operation failed: {e}")

    async def search_async(
        self,
        insurance_name: str,
        provider_name: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Async version of search."""
        try:
            embedding_literal = None
            if query:
                emb = await self.embed_async(query)
                embedding_literal = _format_vector(emb)

            filters = ["insurance_name ILIKE %s"]
            params: List[Any] = [f"%{insurance_name}%"]
            if provider_name:
                filters.append("provider_name ILIKE %s")
                params.append(f"%{provider_name}%")

            filters.append("(chunk_index = 0 OR chunk_index IS NULL)")

            where_clause = " AND ".join(filters)
            order_clause = "ORDER BY created_at DESC"
            if embedding_literal:
                order_clause = "ORDER BY embedding <-> %s::vector"
                params.append(embedding_literal)

            params.append(limit)

            sql = f"""
                SELECT id, insurance_name, provider_name, call_id, request_id,
                       summary, qa_text, metadata, created_at
                FROM call_knowledge
                WHERE {where_clause}
                {order_clause}
                LIMIT %s
            """

            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                logger.debug(f"Async search returned {len(rows)} results")
                return [dict(r) for r in rows]

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            raise SearchError(f"Search operation failed: {e}")
