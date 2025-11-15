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


@dataclass
class SearchResources:
    collection: chromadb.Collection
    page_id_to_title: dict[str, str]
    page_id_to_content: dict[str, str]


def load_environment(
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "https://api.openai.com/v1",
    embed_api_key_var: str = "OPENAI_API_KEY",
    corpus_dataset: str = "willcb/rare-wiki-pages",
    corpus_split: str = "train",
    chroma_db_dir: str = CHROMA_DB_DIR,
) -> SearchResources:
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name=embed_model,
        api_base=embed_base_url,
        api_key=os.getenv(embed_api_key_var, "EMPTY"),
    )
    client = chromadb.PersistentClient(path=chroma_db_dir)
    collection = client.get_or_create_collection(
        name="wiki_titles",
        embedding_function=cast(EmbeddingFunction[Embeddable], openai_ef),
    )

    corpus = load_dataset(corpus_dataset, split=corpus_split)
    page_id_to_title: dict[str, str] = {}
    page_id_to_content: dict[str, str] = {}
    for row in corpus:
        row = cast(dict, row)
        pid = row["id"]
        title = row["title"]
        content = row["content"]
        page_id_to_title[pid] = title
        page_id_to_content[pid] = content

    def init_chroma() -> None:
        all_ids = list(page_id_to_title.keys())
        existing: set[str] = set()
        for i in range(0, len(all_ids), 500):
            batch = all_ids[i : i + 500]
            got = collection.get(ids=batch)
            existing.update(got.get("ids", []))
        missing = [pid for pid in all_ids if pid not in existing]
        if missing:
            documents = []
            metadatas = []
            for pid in missing:
                title = str(page_id_to_title[pid]).strip()
                if not title:
                    raise ValueError(f"Empty title for page_id {pid}")
                documents.append(title)
                metadatas.append({"title": title})
            bs = 100
            for i in range(0, len(missing), bs):
                collection.upsert(
                    ids=missing[i : i + bs],
                    documents=documents[i : i + bs],
                    metadatas=metadatas[i : i + bs],
                )

    init_chroma()

    return SearchResources(
        collection=collection,
        page_id_to_title=page_id_to_title,
        page_id_to_content=page_id_to_content,
    )


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


async def search_pages(resources: SearchResources, query: str) -> list[dict]:
    async with _get_chroma_semaphore():
        results = await asyncio.to_thread(
            resources.collection.query, query_texts=[query], n_results=10
        )
    if not results:
        raise ValueError(f"No results found for query: {query}")
    if not results["metadatas"]:
        raise ValueError(f"No results metadata found for query: {query}")
    output = []
    for i in range(len(results["ids"][0])):
        output.append(
            {
                "page_id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
            }
        )
    return output


async def view_sections(resources: SearchResources, page_id: str) -> list[dict]:
    content = resources.page_id_to_content[page_id]
    sections = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("#"):
            section_name = line.lstrip("#").strip()
            section_id = f"{page_id}:{normalize_id(section_name)}"
            sections.append(
                {
                    "section_id": section_id,
                    "section_name": section_name,
                    "start_line": i,
                }
            )
    if not sections:
        sections.append(
            {
                "section_id": f"{page_id}:full",
                "section_name": "Full Page",
                "start_line": 0,
            }
        )
    return [
        {"section_id": s["section_id"], "section_name": s["section_name"]}
        for s in sections
    ]


async def read_section(resources: SearchResources, section_id: str) -> str:
    if ":" not in section_id:
        raise ValueError("Invalid section_id format. Expected: page_id:section_name")
    page_id, section_name_id = section_id.split(":", 1)
    content = resources.page_id_to_content[page_id]
    lines = content.split("\n")
    if section_name_id == "full":
        return content
    section_start = None
    section_end = None
    for i, line in enumerate(lines):
        if line.startswith("#"):
            current_section = normalize_id(line.lstrip("#").strip())
            if current_section == section_name_id and section_start is None:
                section_start = i
            elif section_start is not None and section_end is None:
                section_end = i
                break
    if section_start is not None:
        if section_end is None:
            section_end = len(lines)
        return "\n".join(lines[section_start:section_end])
    else:
        raise ValueError(f"Section not found: {section_id}")


def _batch(iterable: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def train_with_trl_grpo(
    model_name: str = "gpt2",
    output_dir: str = "./trl-grpo-checkpoints",
    max_steps: int = 10,
    batch_size: int = 2,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    wandb_project: str = "patent-search",
    wandb_run_name: str | None = None,
) -> None:
    # Optional Weights & Biases setup (reads .env wandb_api_key)
    try:
        from dotenv import load_dotenv
        import wandb  # type: ignore
        load_dotenv()
        _wandb_key = os.getenv("WANDB_API_KEY") or os.getenv("wandb_api_key")
        if _wandb_key:
            wandb.login(key=_wandb_key)
        wandb.init(project=wandb_project, name=wandb_run_name, config={
            "model_name": model_name,
            "trainer": "GRPO",
            "batch_size": batch_size,
            "max_steps": max_steps,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        })
        _WANDB = True
    except Exception:
        _WANDB = False

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    grpo_config = GRPOConfig(
        learning_rate=1e-5,
        bf16=True,
    )

    # Some TRL versions require reward_funcs at init; we supply a no-op
    # reward func and still pass explicit rewards to trainer.step below.
    def _noop_reward_func(**kwargs):
        candidates = (
            kwargs.get("samples"),
            kwargs.get("outputs"),
            kwargs.get("responses"),
            kwargs.get("completions"),
            kwargs.get("texts"),
        )
        n = None
        for v in candidates:
            if isinstance(v, (list, tuple)):
                n = len(v)
                break
        if n is None:
            n = 1
        return [0.0] * n

    trainer = GRPOTrainer(args=grpo_config, model=model, reward_funcs=[_noop_reward_func])

    ds = load_dataset("willcb/wiki-trivia-questions-v4", split="train")
    questions: list[str] = [cast(dict, r)["question"] for r in ds]
    answers: list[str] = [cast(dict, r)["answer"] for r in ds]

    prompts = [f"Question: {q}\nAnswer:" for q in questions]

    step = 0
    cursor = 0
    for group in _batch(prompts, batch_size):
        if step >= max_steps:
            break
        group_answers = answers[cursor : cursor + len(group)]

        query_tensors = [
            tokenizer(g, return_tensors="pt", padding=False, truncation=True)[
                "input_ids"
            ][0].to(trainer.accelerator.device)
            for g in group
        ]

        response_tensors = []
        response_texts = []
        for q in query_tensors:
            gen = trainer.generate(
                q.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full = gen[0]
            resp = full[q.size(0) :]
            response_tensors.append(resp)
            response_texts.append(
                tokenizer.decode(resp, skip_special_tokens=True).strip()
            )

        rewards = []
        for resp_txt, ans in zip(response_texts, group_answers):
            reward = 1.0 if ans.lower() in resp_txt.lower() else 0.0
            rewards.append(reward)

        stats = trainer.step(query_tensors, response_tensors, rewards)
        if _WANDB:
            try:
                import wandb  # type: ignore
                wandb.log({
                    "train/step": step,
                    "reward/mean": float(sum(rewards) / max(1, len(rewards))),
                    "reward/std": float((sum((r - (sum(rewards)/max(1,len(rewards))))**2 for r in rewards) / max(1, len(rewards))) ** 0.5),
                    **{f"stats/{k}": (float(v) if isinstance(v, (int, float)) else v) for k, v in (stats or {}).items()},
                })
            except Exception:
                pass
        step += 1
        cursor += len(group)

    trainer.save_pretrained(output_dir)
