import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import tiktoken
import torch
from chonkie import SDPMChunker
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.models.chunk import Chunk

load_dotenv(override=True)

SYSTEM_INSTRUCTION = """You're an expert contextualizer that adds context to chunks to help create a better knowledge graph and search retrieval.
Given a document and a chunk, your job is to situate the chunk by providing short succinct context."""

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

MODEL = "gpt-4.1-mini"

logger = logging.getLogger(__name__)


def get_token_counter(model: str = "gpt-4o-mini"):
    """Get a token counter function for the specified model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return lambda text: len(encoding.encode(text))
    except Exception:
        # Fallback to a simple word-based estimation
        return lambda text: len(text.split())


class SemanticTextSplitter:
    def __init__(self, chunk_size: int = 2048, model: str = "gpt-4o-mini", use_contextual_chunks: bool = True):
        """
        Initialize the text splitter with chunk size and model name parameters.

        Args:
            chunk_size (int): Maximum number of tokens per chunk
            model (str): Model name for token counting
            use_contextual_chunks (bool): Whether to add context to chunks using OpenAI
        """
        self.chunk_size = chunk_size
        self.use_contextual_chunks = use_contextual_chunks
        self.model = model

        # Determine the best device for the embedding model
        # Use CUDA if available, otherwise CPU (avoid MPS for now due to embedding_bag issues)
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA for semantic chunking embeddings")
        elif torch.backends.mps.is_available():
            device = "cpu"  # Force CPU on Apple Silicon due to embedding_bag not implemented for MPS
            logger.info("Using CPU for semantic chunking embeddings (MPS not supported for embedding_bag)")
        else:
            device = "cpu"
            logger.info("Using CPU for semantic chunking embeddings")

        self.chunker = SDPMChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold="auto",
            chunk_size=chunk_size,
            min_sentences=20,
            skip_window=1,
            device=device,  # Explicitly set device
        )

        # Initialize OpenAI client only if contextual chunks are enabled
        if self.use_contextual_chunks:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                logger.warning("OPENAI_API_KEY not found. Contextual chunking will be disabled.")
                self.use_contextual_chunks = False
                self.client = None
        else:
            self.client = None

        self.token_counter = get_token_counter(model)

    def split(self, text: str) -> List[Chunk]:
        """
        Split text into chunks based on both token count and semantic boundaries.
        """
        semantic_chunks = self.chunker.chunk(text)
        logger.info(f"Found {len(semantic_chunks)} semantic chunks")

        # If contextual chunks are disabled or client is not available, return simple chunks
        if not self.use_contextual_chunks or not self.client:
            return [Chunk(content=c.text, metadata={}) for c in semantic_chunks]

        # Use ThreadPoolExecutor to parallelize the situate_chunk calls
        with ThreadPoolExecutor(max_workers=7) as executor:
            # Submit all tasks to the thread pool
            future_to_chunk = {
                executor.submit(self.situate_chunk, chunk.text, text): chunk for chunk in semantic_chunks
            }

            # Collect results in order
            contextualized_chunks = []
            for chunk in semantic_chunks:
                # Find the future corresponding to this chunk
                future = next(f for f, c in future_to_chunk.items() if c == chunk)
                try:
                    context = future.result()
                    # Add context as a prefix to the chunk
                    content_with_context = f"{context}\n\n{chunk.text}"
                    contextualized_chunks.append(Chunk(content=content_with_context, metadata={"has_context": True}))
                except Exception as e:
                    logger.error(f"Failed to add context to chunk: {e}")
                    # Fall back to chunk without context
                    contextualized_chunks.append(Chunk(content=chunk.text, metadata={"has_context": False}))

        return contextualized_chunks

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def situate_chunk(self, chunk: str, document: str) -> str:
        """Add contextual information to a chunk using OpenAI."""
        try:
            # For smaller documents, include the full document in the prompt
            if self.token_counter(document) < 8000:  # Leave room for response
                messages = [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {
                        "role": "user",
                        "content": f"{DOCUMENT_CONTEXT_PROMPT.format(doc_content=document)}\n\n{CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)}",
                    },
                ]
            else:
                # For larger documents, just provide the chunk and ask for general context
                messages = [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {
                        "role": "user",
                        "content": f"Here is a chunk from a larger document:\n<chunk>\n{chunk}\n</chunk>\n\nPlease provide a short succinct context that would help situate this chunk for search retrieval. Answer only with the context and nothing else.",
                    },
                ]

            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=200,  # Keep context short
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to situate chunk: {e}")
            # Return empty context on failure
            return ""
