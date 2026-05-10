import argparse
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chz
import chromadb
import pandas as pd
import tinker
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tinker.types import LossFnType
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.train import AsyncConfig, Config as RLConfig, main as rl_main
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils import logtree

from tinker_logging import (
    append_tool_trace,
    create_run_artifacts,
    disable_cookbook_trajectory_logging,
    finalize_run,
    load_json_config,
    make_run_id,
    set_tool_trace_file,
    summarize,
)


DEFAULT_CHROMA_DIR = ".chroma_db"
DEFAULT_COLLECTION = "patent_collection"
DEFAULT_TRAINING_LOGS_DIR = "training_logs"
DEFAULT_CONFIG_PATH = "prior_art_search/TINKER_grpo_train.config.json"


def canonical_pub_id(pub_id: str) -> str:
    return str(pub_id).strip().split("-")[0]


def format_messages(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"].upper()
        if role == "TOOL":
            name = message.get("name", "tool")
            chunks.append(f"TOOL({name}):\n{message['content']}")
        else:
            chunks.append(f"{role}:\n{message['content']}")
    chunks.append("ASSISTANT:")
    return "\n\n".join(chunks)


def first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class PatentTools:
    """
    Uses an existing local Chroma collection.
    Does not create/populate data.
    """

    def __init__(self, chroma_dir: str = DEFAULT_CHROMA_DIR, collection_name: str = DEFAULT_COLLECTION):
        client = chromadb.PersistentClient(path=chroma_dir)
        try:
            self.collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                ),
            )
        except Exception as exc:
            available = [c.name for c in client.list_collections()]
            raise RuntimeError(
                f"Collection '{collection_name}' not found in '{chroma_dir}'. "
                f"Available collections: {available}"
            ) from exc

        ids = self.collection.get(include=[])["ids"]
        self.ids_by_canonical: dict[str, str] = {}
        for patent_id in ids:
            self.ids_by_canonical.setdefault(canonical_pub_id(patent_id), patent_id)

    def search_patents(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        out: list[dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            out.append(
                {
                    "patent_title": results["metadatas"][0][i].get("title", ""),
                    "publication_number": results["ids"][0][i],
                    "similarity_score": float(results["distances"][0][i]),
                }
            )
        return out

    def lookup_patent(self, publication_number: str) -> dict[str, Any]:
        result = self.collection.get(ids=[publication_number])
        if result and result.get("metadatas"):
            metadata = result["metadatas"][0] or {}
            search_text = (result.get("documents") or [""])[0]
            return {**metadata, "search_text": search_text}

        canonical = canonical_pub_id(publication_number)
        fallback_id = self.ids_by_canonical.get(canonical)
        if not fallback_id:
            raise ValueError(f"No patent found for {publication_number}")

        result = self.collection.get(ids=[fallback_id])
        if not result or not result.get("metadatas"):
            raise ValueError(f"No patent found for {publication_number}")
        metadata = result["metadatas"][0] or {}
        search_text = (result.get("documents") or [""])[0]
        return {**metadata, "search_text": search_text}


@dataclass
class PatentScenario:
    publication_number: str
    query: str

class PatentSearchEnv(Env):
    def __init__(
        self,
        scenario: PatentScenario,
        tools: PatentTools,
        tokenizer: Any,
        max_turns: int,
    ):
        self.scenario = scenario
        self.tools = tools
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn = 0
        self.final_reward = 0.0

        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    "Query:\n"
                    f"{self.scenario.query}\n\n"
                    "Find relevant prior patents. Return final answer with publication numbers."
                ),
            },
        ]

    @property
    def stop_condition(self) -> StopCondition:
        return []

    def _system_prompt(self) -> str:
        return (
            "You are a prior-art patent search agent.\n"
            f"You may take up to {self.max_turns} turns.\n"
            "At each turn output:\n"
            "1) one short sentence prefixed with 'Thought:'\n"
            "2) then exactly one JSON object:\n"
            '{"tool":"search_patents","arguments":{"query":"...","n_results":10}}\n'
            '{"tool":"lookup_patent","arguments":{"publication_number":"..."}}\n'
            '{"tool":"return_final_answer","answer":"...","patent_ids":["..."]}\n'
            "Do not use markdown fences."
        )

    def _build_observation(self) -> Observation:
        prompt = format_messages(self.messages)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return tinker.ModelInput.from_ints(tokens=tokens)

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self._build_observation(), self.stop_condition

    async def step(self, action: list[int]) -> StepResult:
        self.turn += 1
        assistant_text = self.tokenizer.decode(action).strip()
        self.messages.append({"role": "assistant", "content": assistant_text})
        logtree.log_text(f"Turn {self.turn} assistant:\n{assistant_text}")

        reward = 0.0
        done = False
        metrics: Metrics = {"turn": self.turn}
        args: dict[str, Any] | None = None
        tool_result: Any = None

        parsed = first_json_object(assistant_text)
        tool_name = ""
        if parsed is None or not isinstance(parsed.get("tool"), str):
            done = True
            metrics["parse_ok"] = 0.0
        else:
            metrics["parse_ok"] = 1.0
            tool_name = parsed["tool"]
            args = parsed.get("arguments", {}) if isinstance(parsed.get("arguments"), dict) else {}
            tool_result = {}

            try:
                if tool_name == "search_patents":
                    query = str(args.get("query", "")).strip()
                    n_results = int(args.get("n_results", 10))
                    tool_result = await asyncio.to_thread(
                        self.tools.search_patents, query=query, n_results=n_results
                    )
                elif tool_name == "lookup_patent":
                    pub = str(args.get("publication_number", "")).strip()
                    tool_result = await asyncio.to_thread(self.tools.lookup_patent, pub)
                elif tool_name == "return_final_answer":
                    raw_ids = parsed.get("patent_ids", [])
                    ids = [str(x) for x in raw_ids] if isinstance(raw_ids, list) else []
                    gold = canonical_pub_id(self.scenario.publication_number)
                    predicted = {canonical_pub_id(x) for x in ids}
                    reward = 1.0 if gold in predicted else 0.0
                    done = True
                    metrics["correct"] = reward
                    metrics["num_returned_ids"] = len(ids)
                    tool_result = {"reward": reward, "gold": gold, "predicted": list(predicted)}
                else:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
            except Exception as exc:
                tool_result = {"error": str(exc)}

            if not done:
                self.messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=True),
                    }
                )

            logtree.log_text(
                f"Turn {self.turn} tool={tool_name}\nresult={summarize(tool_result)}"
            )

        if not done and self.turn >= self.max_turns:
            done = True
            metrics["max_turn_reached"] = 1.0

        if done:
            self.final_reward = reward
            next_ob = tinker.ModelInput.empty()
        else:
            next_ob = self._build_observation()

        metrics["reward"] = reward
        append_tool_trace(
            publication_number=self.scenario.publication_number,
            query=self.scenario.query,
            turn=self.turn,
            assistant_text=assistant_text,
            parsed_action=parsed,
            tool_name=tool_name or None,
            tool_args=args,
            tool_result=tool_result,
            reward=reward,
            done=done,
            metrics=metrics,
        )
        return StepResult(
            reward=reward,
            episode_done=done,
            next_observation=next_ob,
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


@dataclass(frozen=True)
class PatentEnvGroupBuilder(EnvGroupBuilder):
    scenario: PatentScenario
    tools: PatentTools
    tokenizer: Any
    group_size: int
    max_turns: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            PatentSearchEnv(
                scenario=self.scenario,
                tools=self.tools,
                tokenizer=self.tokenizer,
                max_turns=self.max_turns,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["patent", "prior_art", "tool_use"]


class PatentRLDataset(RLDataset):
    def __init__(
        self,
        scenarios: list[PatentScenario],
        tools: PatentTools,
        tokenizer: Any,
        groups_per_batch: int,
        group_size: int,
        max_turns: int,
        n_batches: int,
        seed: int,
    ):
        if not scenarios:
            raise ValueError("PatentRLDataset requires at least one scenario.")
        self.scenarios = scenarios
        self.tools = tools
        self.tokenizer = tokenizer
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.max_turns = max_turns
        self.n_batches = n_batches
        self.order = list(range(len(scenarios)))
        random.Random(seed).shuffle(self.order)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = (index * self.groups_per_batch) % len(self.order)
        builders: list[EnvGroupBuilder] = []
        for i in range(self.groups_per_batch):
            scenario_idx = self.order[(start + i) % len(self.order)]
            builders.append(
                PatentEnvGroupBuilder(
                    scenario=self.scenarios[scenario_idx],
                    tools=self.tools,
                    tokenizer=self.tokenizer,
                    group_size=self.group_size,
                    max_turns=self.max_turns,
                )
            )
        return builders

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class PatentDatasetBuilder(RLDatasetBuilder):
    csv_path: str
    model_name_for_tokenizer: str
    chroma_dir: str
    collection_name: str
    groups_per_batch: int
    group_size: int
    max_turns: int
    train_batches: int
    eval_batches: int
    seed: int

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        csv_file = Path(self.csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_file}")

        df = pd.read_csv(csv_file)
        if len(df) < 10:
            raise ValueError("Dataset too small for train/eval split.")

        shuffled = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        split_idx = int(len(shuffled) * 0.8)
        train_df = shuffled.iloc[:split_idx]
        eval_df = shuffled.iloc[split_idx:]

        train_scenarios = [
            PatentScenario(
                publication_number=str(row["publication_number"]),
                query=str(row["query"]),
            )
            for row in train_df.to_dict(orient="records")
        ]
        eval_scenarios = [
            PatentScenario(
                publication_number=str(row["publication_number"]),
                query=str(row["query"]),
            )
            for row in eval_df.to_dict(orient="records")
        ]

        tools = PatentTools(chroma_dir=self.chroma_dir, collection_name=self.collection_name)
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)

        train_dataset = PatentRLDataset(
            scenarios=train_scenarios,
            tools=tools,
            tokenizer=tokenizer,
            groups_per_batch=self.groups_per_batch,
            group_size=self.group_size,
            max_turns=self.max_turns,
            n_batches=self.train_batches,
            seed=self.seed,
        )

        eval_dataset: RLDataset | None = None
        if self.eval_batches > 0 and eval_scenarios:
            eval_dataset = PatentRLDataset(
                scenarios=eval_scenarios,
                tools=tools,
                tokenizer=tokenizer,
                groups_per_batch=self.groups_per_batch,
                group_size=self.group_size,
                max_turns=self.max_turns,
                n_batches=self.eval_batches,
                seed=self.seed + 1,
            )

        return train_dataset, eval_dataset


def parse_args() -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    config_defaults = load_json_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description="Patent prior-art RL training using official tinker_cookbook RL loop."
    )
    parser.add_argument("--config", default=bootstrap_args.config)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--loss-fn", default="ppo", choices=["importance_sampling", "ppo"])
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--groups-per-batch", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--csv-path", default="Evals/patent_search_queries.csv")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--training-logs-dir", default=DEFAULT_TRAINING_LOGS_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--log-path", default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remove-constant-reward-groups", action="store_true")
    parser.add_argument("--enable-trace", action="store_true")
    parser.add_argument("--num-groups-to-log", type=int, default=0)
    parser.add_argument("--max-steps-off-policy", type=int, default=-1)
    if config_defaults:
        valid_dests = {action.dest for action in parser._actions}
        unknown_keys = sorted(k for k in config_defaults.keys() if k not in valid_dests)
        if unknown_keys:
            raise ValueError(
                f"Unknown config keys in {bootstrap_args.config}: {unknown_keys}"
            )
        parser.set_defaults(**config_defaults)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv()
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("Missing TINKER_API_KEY.")

    run_id = args.run_id or make_run_id(args.model)
    artifacts = create_run_artifacts(
        model_name=args.model,
        training_logs_dir=args.training_logs_dir,
        log_path=args.log_path,
        run_id=run_id,
    )

    run_metadata = {
        "run_id": artifacts.run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "loss_fn": args.loss_fn,
        "steps": args.steps,
        "groups_per_batch": args.groups_per_batch,
        "group_size": args.group_size,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "learning_rate": args.learning_rate,
        "chroma_dir": args.chroma_dir,
        "collection_name": args.collection_name,
        "num_groups_to_log": args.num_groups_to_log,
        "run_dir": str(artifacts.run_dir),
        "metrics_dir": str(artifacts.metrics_dir),
        "traces_dir": str(artifacts.traces_dir),
        "logs_file": str(artifacts.logs_file),
        "tool_trace_file": str(artifacts.tool_trace_file),
        "wandb_project": args.wandb_project or None,
        "wandb_name": args.wandb_name or None,
    }
    (artifacts.run_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    dataset_builder = PatentDatasetBuilder(
        csv_path=args.csv_path,
        model_name_for_tokenizer=args.model,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        max_turns=args.max_turns,
        train_batches=args.steps,
        eval_batches=args.eval_batches,
        seed=args.seed,
    )

    async_config = None
    if args.max_steps_off_policy >= 0:
        async_config = AsyncConfig(
            max_steps_off_policy=args.max_steps_off_policy,
            groups_per_batch=args.groups_per_batch,
        )

    cfg = RLConfig(
        learning_rate=args.learning_rate,
        dataset_builder=dataset_builder,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lora_rank=args.rank,
        log_path=str(artifacts.run_dir),
        wandb_project=args.wandb_project or None,
        wandb_name=args.wandb_name or None,
        base_url=args.base_url or None,
        eval_every=args.eval_every,
        save_every=args.save_every,
        remove_constant_reward_groups=args.remove_constant_reward_groups,
        enable_trace=args.enable_trace,
        num_groups_to_log=args.num_groups_to_log,
        loss_fn=args.loss_fn,  # ppo / importance_sampling from tinker-cookbook
        async_config=async_config,
    )

    print("=== TINKER COOKBOOK RL TRAIN (PATENT) ===")
    print(f"model={args.model} loss_fn={args.loss_fn}")
    print(
        f"groups_per_batch={args.groups_per_batch} group_size={args.group_size} "
        f"steps={args.steps}"
    )
    print(f"chroma_dir={args.chroma_dir} collection={args.collection_name}")
    print(f"run_dir={artifacts.run_dir}")
    print(f"metrics_dir={artifacts.metrics_dir}")
    print(f"traces_dir={artifacts.traces_dir}")
    print(f"logs_file={artifacts.logs_file}")
    print(f"tool_trace_file={artifacts.tool_trace_file}")

    set_tool_trace_file(artifacts.tool_trace_file)
    disable_cookbook_trajectory_logging()
    try:
        await rl_main(cfg)
    finally:
        set_tool_trace_file(None)

    finalize_run(
        artifacts=artifacts,
        training_logs_dir=args.training_logs_dir,
        run_metadata=run_metadata,
    )


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
