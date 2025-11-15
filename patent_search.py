import asyncio
import os
from dataclasses import dataclass
from typing import cast, Iterable, Sequence

import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import GRPOConfig, GRPOTrainer

CHROMA_DB_DIR = ".chroma_db"
_chroma_semaphore: asyncio.Semaphore | None = None


def _get_chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(100)
    return _chroma_semaphore

