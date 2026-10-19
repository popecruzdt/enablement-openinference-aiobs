"""
RAG (Retrieval-Augmented Generation) Pipeline for Travel Advisor
Handles document loading, embedding, and retrieval-augmented generation
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama

import httpx
import weaviate
import weaviate.classes as wvc
from weaviate.config import AdditionalConfig, Timeout

from src.config import get_settings
from src.feature_flags import get_embedding_override, set_embedding_override
from src.telemetry.langchain_context import langchain_span_context

logger = logging.getLogger(__name__)


def _set_genai_request_attributes(model_name: str) -> None:
    """Attach GenAI semantic attributes to the current span when available."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("gen_ai.response.model", model_name)
            span.set_attribute("gen_ai.system", "ollama")
    except Exception:
        # Tracing must never affect retrieval/generation behavior.
        pass


DEFAULT_RAG_PROMPT_TEMPLATE = """You are a helpful travel advisor assistant. Use ONLY the information provided in the context below to answer questions about travel destinations.

CRITICAL INSTRUCTIONS:
- Use ONLY the facts from the context provided below
- Do NOT use any external knowledge or information you may have
- If the context contains information about the location, use it exactly as written
- If the context does not contain relevant information, say \"I don't have information about that destination\"
- Keep your response concise and helpful, maximum 50 words

<context>
{context}
</context>

Question: {destination}

Travel Advice:"""


@dataclass
class RetrievalResult:
    """Destination context retrieved from the Weaviate KB."""
    contexts: List[str] = field(default_factory=list)
    drift_active: bool = False
    destination_found: bool = False


class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline
    Manages document indexing and retrieval-augmented generation
    """

    def __init__(self):
        """Initialize the RAG pipeline with configuration"""
        self.settings = get_settings()
        self.llm = ChatOllama(
            model=self.settings.ai_model,
            base_url=self.settings.ollama_endpoint,
            temperature=self.settings.ai_temperature
        )
        self.weaviate_client: Optional[weaviate.Client] = None
        self.rag_chain = None

    def connect_weaviate(self) -> weaviate.Client:
        """
        Connect to Weaviate vector database
        
        Returns:
            weaviate.Client: Connected Weaviate client
            
        Raises:
            Exception: If connection fails
        """
        try:
            if self.weaviate_client is None:
                url = f"{self.settings.weaviate_scheme}://{self.settings.weaviate_endpoint}:{self.settings.weaviate_port}"
                logger.info(f"Connecting to Weaviate at {url}")
                
                if self.settings.weaviate_scheme == "http":
                    self.weaviate_client = weaviate.connect_to_local(
                        host=self.settings.weaviate_endpoint,
                        port=self.settings.weaviate_port,
                        additional_config=AdditionalConfig(
                            timeout=Timeout(init=2, query=45, insert=300)
                        )
                    )
                else:
                    self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=url,
                        auth_credentials=weaviate.auth.AuthApiKey("dummy-key")
                    )
                
                logger.info("Connected to Weaviate successfully")
            return self.weaviate_client
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def reset_weaviate_connection(self) -> None:
        """Close the current Weaviate client so the next call creates a fresh connection."""
        if self.weaviate_client is not None:
            try:
                self.weaviate_client.close()
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {e}")
            finally:
                self.weaviate_client = None

    def _format_docs(self, docs) -> str:
        """
        Format documents for prompt inclusion
        
        Args:
            docs: List of document objects
            
        Returns:
            str: Formatted document content
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def _select_best_exact_match(self, destination: str, objects) -> Optional[str]:
        """Select the best exact-match context for a destination query."""
        normalized_destination = destination.strip().lower()

        # First preference: collect ALL chunks where title exactly matches destination
        texts = []
        for obj in objects:
            props = obj.properties or {}
            title = (props.get("title") or "").strip().lower()
            text = props.get("text") or ""
            if title == normalized_destination and text:
                texts.append(text)

        if texts:
            return "\n\n".join(texts)

        # Second preference: destination term appears in document text.
        for obj in objects:
            props = obj.properties or {}
            text = props.get("text") or ""
            if normalized_destination in text.lower() and text:
                return text

        return None

    def _set_weaviate_common_attributes(
        self,
        span,
        operation: str,
        destination: str,
        attempt: int,
    ) -> None:
        """Attach common context attributes to manual Weaviate spans."""
        if span is None or not span.is_recording():
            return
        span.set_attribute("db.system", "weaviate")
        span.set_attribute("server.address", self.settings.weaviate_endpoint)
        span.set_attribute("server.port", int(self.settings.weaviate_port))
        span.set_attribute("db.operation.name", operation)
        span.set_attribute("weaviate.collection", "KB")
        span.set_attribute("rag.destination", destination)
        span.set_attribute("rag.retry.attempt", attempt)
        span.set_attribute("gen_ai.operation.name", "retrieve")

    def load_rag_prompt_template(self) -> str:
        """Load the RAG prompt template from the filesystem, with a code fallback."""
        prompt_path = Path(self.settings.rag_prompt_path)
        if prompt_path.exists():
            logger.info(f"Loading RAG prompt template from {prompt_path}")
            return prompt_path.read_text(encoding="utf-8")

        logger.warning(
            f"RAG prompt template file not found at {prompt_path}; using built-in fallback template"
        )
        return DEFAULT_RAG_PROMPT_TEMPLATE

    def prepare_knowledge_base(self, destinations_path: Optional[str] = None) -> None:
        """
        Prepare the knowledge base by loading and indexing destination documents
        
        Args:
            destinations_path: Path to destinations directory (uses config if None)
        """
        if destinations_path is None:
            destinations_path = self.settings.destinations_path

        client = self.connect_weaviate()

        # Skip re-indexing if KB is already populated and force_reindex is not set
        if not self.settings.force_reindex:
            try:
                existing_kb = client.collections.get("KB")
                agg = existing_kb.aggregate.over_all(total_count=True)
                if (agg.total_count or 0) >= self.settings.min_kb_objects:
                    logger.info(
                        f"KB already populated ({agg.total_count} objects). "
                        "Skipping re-indexing. Set FORCE_REINDEX=true to rebuild."
                    )
                    return
            except Exception:
                pass  # Collection doesn't exist yet — fall through to create it

        # Clean up existing collection
        try:
            logger.info("Cleaning up existing 'KB' collection...")
            client.collections.delete("KB")
        except Exception as e:
            logger.warning(f"Could not delete existing collection: {e}")

        # Create new collection
        logger.info("Creating new 'KB' collection...")
        client.collections.create(
            name="KB",
            vector_config=wvc.config.Configure.Vectors.text2vec_ollama(
                name="default",
                source_properties=["title", "text"],
                api_endpoint=self.settings.ollama_endpoint,
                model=self.settings.ai_embedding_model
            ),
            properties=[
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="source",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="title",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
        )

        # Load and process documents
        docs_list = []
        if os.path.exists(destinations_path):
            logger.info(f"Loading documents from {destinations_path}")
            for filename in os.listdir(destinations_path):
                if filename.endswith(".html"):
                    file_path = os.path.join(destinations_path, filename)
                    logger.info(f"  Loading {filename}...")
                    try:
                        loader = BSHTMLLoader(file_path=file_path)
                        item_docs = loader.load()
                        docs_list.extend(item_docs)
                    except Exception as e:
                        logger.warning(f"Error loading {filename}: {e}")
        else:
            logger.warning(f"Destinations path does not exist: {destinations_path}")

        if not docs_list:
            logger.warning("No documents loaded. Knowledge base will be empty.")
            return

        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
        documents = text_splitter.split_documents(docs_list)
        logger.info(f"Created {len(documents)} document chunks")

        # Index documents with Weaviate v4 client
        logger.info("Indexing documents in Weaviate...")
        kb = client.collections.get("KB")
        indexed_count = 0
        try:
            with kb.batch.fixed_size(batch_size=10) as batch:
                for doc in documents:
                    source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
                    title = Path(source).stem if source else ""
                    batch.add_object(
                        properties={
                            "text": doc.page_content,
                            "source": source,
                            "title": title,
                        }
                    )
                    indexed_count += 1
                # Batch auto-flushes when exiting context
            
            # Verify objects were actually stored
            agg = kb.aggregate.over_all(total_count=True)
            logger.info(f"Batch indexing complete. Total objects in KB collection: {agg.total_count}")
            if agg.total_count == 0:
                raise RuntimeError("No objects found in KB after indexing. Weaviate accepted the schema but stored no documents.")
            
        except Exception as e:
            logger.error(f"Error during batch indexing: {e}", exc_info=True)
            raise
        
        logger.info("Knowledge base prepared successfully")

    def set_embedding_model(self, model: str) -> dict:
        """
        Set or clear the embedding model override feature flag.
        When overridden to a mismatched model, near_text queries will produce a
        dimension mismatch against stored 768-dim vectors, simulating embedding drift.
        To restore: call with the configured AI_EMBEDDING_MODEL value.

        Args:
            model: Ollama model name (e.g. 'gemma2:2b' to break, 'nomic-embed-text' to fix)

        Returns:
            dict with previous_model, current_model, and collection name
        """
        previous = get_embedding_override() or self.settings.ai_embedding_model
        reset = (model == self.settings.ai_embedding_model)
        set_embedding_override("" if reset else model)
        logger.info(f"Embedding model override: {previous} -> {model}")
        return {"previous_model": previous, "current_model": model, "collection": "KB"}

    def _compute_embedding(self, text: str, model: str) -> list:
        """
        Compute an embedding vector for the given text using a specific Ollama model.
        Used when the embedding-model-override feature flag is active.

        Args:
            text: The text to embed
            model: Ollama model name to use for embedding

        Returns:
            list of floats representing the embedding vector
        """
        resp = httpx.post(
            f"{self.settings.ollama_endpoint}/api/embed",
            json={"model": model, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        raw_vec = resp.json()["embeddings"][0]

        # The KB collection was indexed with gemma2:2b (2304-dim vectors via text2vec-ollama).
        # If the override model returns a vector with a DIFFERENT dimension, we deliberately
        # produce a dimension mismatch so that Weaviate raises an exception on near_vector.
        # The caller catches that exception and records distance=1.0 on the OTel span,
        # creating the observable drift anomaly in Dynatrace.
        # We use 768 as the sentinel "wrong" dimension (nomic-embed-text output size).
        # gemma2:2b returns 2304-dim, so 2304 != 768 → truncate+negate → Weaviate rejects.
        stored_dim = 768  # intentional mismatch sentinel (KB actually stores 2304-dim)
        if len(raw_vec) != stored_dim:
            logger.info(
                f"Dim mismatch: override model '{model}' returned {len(raw_vec)}-dim vector; "
                f"truncating to {stored_dim} dims to force Weaviate dimension error (drift simulation)"
            )
            return [-x for x in raw_vec[:stored_dim]]
        return raw_vec

    def initialize_rag_chain(self) -> None:
        """
        Initialize the RAG retrieval and generation chain
        """
        logger.info("Initializing RAG chain...")
        self.connect_weaviate()
        # Mark chain as ready; retrieval/generation is executed in get_travel_advice.
        self.rag_chain = True
        logger.info("RAG chain initialized successfully")

    def retrieve_context(self, destination: str) -> RetrievalResult:
        """
        Retrieve destination context from the Weaviate KB.
        Shared by the RAG workflow and the agentic workflow's KB search tool.

        Args:
            destination: The destination to retrieve context for

        Returns:
            RetrievalResult: Retrieved context chunks and retrieval flags
        """
        # Inside the agent's KB search tool, nest the weaviate.* spans under the tool span.
        with langchain_span_context():
            return self._retrieve_context(destination)

    def _retrieve_context(self, destination: str) -> RetrievalResult:
        if self.rag_chain is None:
            raise RuntimeError("RAG chain not initialized. Call initialize_rag_chain() first.")

        try:
            tracer = trace.get_tracer("ai-travel-advisor.weaviate")
            results = None
            collection_found = False
            collection_size = 0
            destination_found_in_collection = False
            for attempt in range(1, 3):
                try:
                    with tracer.start_as_current_span("weaviate.connect", kind=SpanKind.CLIENT) as span:
                        self._set_weaviate_common_attributes(span, "connect", destination, attempt)
                        client = self.connect_weaviate()

                    with tracer.start_as_current_span("weaviate.collections.get", kind=SpanKind.CLIENT) as span:
                        self._set_weaviate_common_attributes(span, "collections.get", destination, attempt)
                        kb = client.collections.get("KB")
                    collection_found = True
                    logger.info("Collection found: KB")

                    with tracer.start_as_current_span("weaviate.aggregate.over_all", kind=SpanKind.CLIENT) as span:
                        self._set_weaviate_common_attributes(span, "aggregate.over_all", destination, attempt)
                        agg = kb.aggregate.over_all(total_count=True)
                        span.set_attribute("weaviate.aggregate.total_count", int(agg.total_count or 0))
                    collection_size = agg.total_count or 0
                    logger.info(f"Collection size: {collection_size}")

                    with tracer.start_as_current_span("weaviate.query.bm25", kind=SpanKind.CLIENT) as span:
                        self._set_weaviate_common_attributes(span, "query.bm25", destination, attempt)
                        span.set_attribute("weaviate.query.limit", int(self.settings.retrieval_k))
                        exact_results = kb.query.bm25(
                            query=destination,
                            limit=self.settings.retrieval_k,
                            return_properties=["text", "title"],
                            return_metadata=wvc.query.MetadataQuery(score=True),
                        )
                        span.set_attribute("weaviate.result.count", len(exact_results.objects))
                        bm25_scores = [
                            obj.metadata.score
                            for obj in exact_results.objects
                            if obj.metadata and obj.metadata.score is not None
                        ]
                        if bm25_scores:
                            span.set_attribute("weaviate.result.score.max", round(max(bm25_scores), 6))
                            span.set_attribute("weaviate.result.score.min", round(min(bm25_scores), 6))
                            span.set_attribute("weaviate.result.score.avg", round(sum(bm25_scores) / len(bm25_scores), 6))
                    normalized_destination = destination.strip().lower()
                    destination_found_in_collection = any(
                        ((obj.properties or {}).get("title", "").strip().lower() == normalized_destination)
                        or (normalized_destination in ((obj.properties or {}).get("text", "").lower()))
                        for obj in exact_results.objects
                    )
                    logger.info(f"Destination found in collection: {destination_found_in_collection}")

                    with tracer.start_as_current_span("weaviate.query.near_text", kind=SpanKind.CLIENT) as span:
                        self._set_weaviate_common_attributes(span, "query.near_text", destination, attempt)
                        span.set_attribute("weaviate.query.limit", int(self.settings.retrieval_k))
                        span.set_attribute("weaviate.target_vector", "default")
                        model_override = get_embedding_override()
                        drift_active = False
                        if model_override:
                            logger.info(f"Embedding override active: using near_vector with model '{model_override}'")
                            try:
                                vec = self._compute_embedding(destination, model_override)
                                results = kb.query.near_vector(
                                    near_vector=vec,
                                    target_vector="default",
                                    limit=self.settings.retrieval_k,
                                    return_properties=["text", "source", "title"],
                                    return_metadata=wvc.query.MetadataQuery(distance=True),
                                )
                                span.set_attribute("weaviate.result.count", len(results.objects))
                                distances = [
                                    obj.metadata.distance
                                    for obj in results.objects
                                    if obj.metadata and obj.metadata.distance is not None
                                ]
                                if distances:
                                    span.set_attribute("weaviate.result.distance.min", round(min(distances), 6))
                                    span.set_attribute("weaviate.result.distance.max", round(max(distances), 6))
                                    span.set_attribute("weaviate.result.distance.avg", round(sum(distances) / len(distances), 6))
                            except Exception as drift_exc:
                                logger.warning(f"near_vector failed with override model '{model_override}': {drift_exc}")
                                span.set_attribute("weaviate.result.distance.min", 1.0)
                                span.set_attribute("weaviate.result.distance.avg", 1.0)
                                span.set_attribute("weaviate.result.distance.max", 1.0)
                                span.set_attribute("weaviate.result.count", 0)
                                results = None
                                drift_active = True
                        else:
                            results = kb.query.near_text(
                                query=destination,
                                limit=self.settings.retrieval_k,
                                target_vector="default",
                                return_properties=["text", "source", "title"],
                                return_metadata=wvc.query.MetadataQuery(distance=True),
                            )
                            span.set_attribute("weaviate.result.count", len(results.objects))
                            distances = [
                                obj.metadata.distance
                                for obj in results.objects
                                if obj.metadata and obj.metadata.distance is not None
                            ]
                            if distances:
                                span.set_attribute("weaviate.result.distance.min", round(min(distances), 6))
                                span.set_attribute("weaviate.result.distance.max", round(max(distances), 6))
                                span.set_attribute("weaviate.result.distance.avg", round(sum(distances) / len(distances), 6))
                    break
                except Exception as e:
                    current_span = trace.get_current_span()
                    if current_span is not None and current_span.is_recording():
                        current_span.record_exception(e)
                        current_span.set_status(Status(StatusCode.ERROR, str(e)))
                    if "Deadline Exceeded" in str(e) and attempt == 1:
                        logger.warning("Weaviate near_text timed out; resetting client and retrying once")
                        self.reset_weaviate_connection()
                        continue
                    raise

            contexts = []
            if not drift_active:
                if destination_found_in_collection:
                    best_text = self._select_best_exact_match(destination, exact_results.objects)
                    if best_text:
                        contexts.append(best_text)
                        logger.info("Exact destination lookup detected; using single best-matching document for prompt context")
                    else:
                        logger.warning("Exact destination lookup detected, but no best exact-match text found")

                if not contexts:
                    for obj in results.objects:
                        text = (obj.properties or {}).get("text", "")
                        if text:
                            contexts.append(text)

            if not contexts and not drift_active:
                logger.warning(f"No KB documents retrieved for destination: {destination}")

            return RetrievalResult(
                contexts=contexts,
                drift_active=drift_active,
                destination_found=destination_found_in_collection,
            )
        except Exception as e:
            logger.error(f"Error retrieving destination context: {e}")
            raise

    def get_travel_advice(self, destination: str) -> str:
        """
        Get travel advice for a destination using RAG

        Args:
            destination: The destination to get advice about

        Returns:
            str: Travel advice for the destination
        """
        if self.rag_chain is None:
            raise RuntimeError("RAG chain not initialized. Call initialize_rag_chain() first.")

        logger.info("RAG mode is active")
        logger.info(f"Getting travel advice for: {destination}")
        try:
            retrieval = self.retrieve_context(destination)
            contexts = retrieval.contexts

            if not contexts and not retrieval.drift_active:
                logger.info("Response used RAG: false")
                return "I don't have information about that destination"

            context = "\n\n".join(contexts)  # empty string when drift_active
            logger.info(f"Context contains {len(context)} characters from {len(contexts)} documents")
            
            prompt_template = self.load_rag_prompt_template()
            prompt = prompt_template.format(context=context, destination=destination)

            logger.info(f"Invoking LLM with {len(prompt)} character prompt...")
            _set_genai_request_attributes(self.settings.ai_model)
            response = self.llm.invoke(prompt)
            logger.info(f"LLM returned successfully")
            logger.info("Response used RAG: true")
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Error getting travel advice: {e}")
            raise


def get_rag_pipeline() -> RAGPipeline:
    """
    Factory function to get or create RAG pipeline instance
    
    Returns:
        RAGPipeline: Initialized RAG pipeline
    """
    return RAGPipeline()
