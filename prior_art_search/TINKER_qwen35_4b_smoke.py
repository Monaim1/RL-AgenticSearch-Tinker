import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import tinker
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tinker import types


MAX_TURNS_DEFAULT = 6
COLLECTION_NAME = "patent_collection"
CHROMA_DB_DIR = ".chroma_db"


@dataclass
class TraceStep:
    turn: int
    assistant_text: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result_preview: str | None = None


@dataclass
class RolloutSample:
    prompt_text: str
    answer_text: str
    answer_tokens: list[int]
    answer_logprobs: list[float]
    patent_ids: list[str]
    reward: float
    trace: list[TraceStep] = field(default_factory=list)


class PatentTools:
    def __init__(
        self,
        chroma_dir: str = CHROMA_DB_DIR,
        collection_name: str = COLLECTION_NAME,
        fallback_csv_path: str = "Evals/patent_search_queries.csv",
    ):
        self.mode = "chroma"
        self.collection = None
        self.records_by_pub: dict[str, dict[str, Any]] = {}

        client = chromadb.PersistentClient(path=chroma_dir)
        try:
            self.collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                ),
            )
        except Exception:
            self.mode = "csv_fallback"
            csv_path = Path(fallback_csv_path)
            if not csv_path.exists():
                raise RuntimeError(
                    "Could not open Chroma collection and fallback CSV is missing: "
                    f"{fallback_csv_path}"
                )
            df = pd.read_csv(csv_path)
            # Keep one abstract/query per publication number for lookup/search context.
            dedup = df.drop_duplicates(subset=["publication_number"])
            for row in dedup.to_dict(orient="records"):
                pub = str(row.get("publication_number", "")).strip()
                if not pub:
                    continue
                self.records_by_pub[pub] = {
                    "publication_number": pub,
                    "title": row.get("query", ""),
                    "abstract": row.get("abstract", ""),
                }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}

    def search_patents(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        if self.mode == "chroma":
            results = self.collection.query(query_texts=[query], n_results=n_results)
            output: list[dict[str, Any]] = []
            for i in range(len(results["ids"][0])):
                output.append(
                    {
                        "patent_title": results["metadatas"][0][i].get("title", ""),
                        "publication_number": results["ids"][0][i],
                        "similarity_score": float(results["distances"][0][i]),
                    }
                )
            return output

        q_tokens = self._tokenize(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in self.records_by_pub.values():
            text = f"{rec.get('title', '')} {rec.get('abstract', '')}"
            d_tokens = self._tokenize(text)
            if not d_tokens:
                continue
            overlap = len(q_tokens & d_tokens)
            score = overlap / max(len(q_tokens), 1)
            if overlap > 0:
                scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        output = []
        for score, rec in scored[:n_results]:
            output.append(
                {
                    "patent_title": rec.get("title", ""),
                    "publication_number": rec["publication_number"],
                    "similarity_score": float(1.0 - score),
                }
            )
        return output

    def lookup_patent(self, publication_number: str) -> dict[str, Any]:
        if self.mode == "chroma":
            results = self.collection.get(ids=[publication_number])
            if not results or not results.get("metadatas"):
                raise ValueError(f"No patent found for {publication_number}")
            metadata = results["metadatas"][0] or {}
            abstract = (results.get("documents") or [""])[0]
            return {**metadata, "abstract": abstract}

        record = self.records_by_pub.get(publication_number)
        if not record:
            target = canonical_pub_id(publication_number)
            for pub, rec in self.records_by_pub.items():
                if canonical_pub_id(pub) == target:
                    record = rec
                    break
        if not record:
            raise ValueError(f"No patent found for {publication_number}")
        return dict(record)


def build_system_prompt(max_turns: int) -> str:
    return (
        "You are a prior-art search agent.\n"
        "You must solve the task using iterative tool use.\n"
        f"You can use at most {max_turns} turns.\n\n"
        "At each turn:\n"
        "1) Write one short visible reasoning sentence prefixed with 'Thought:'.\n"
        "2) Then output exactly one JSON object with one of these forms:\n"
        '{"tool":"search_patents","arguments":{"query":"...","n_results":10}}\n'
        '{"tool":"lookup_patent","arguments":{"publication_number":"..."}}\n'
        '{"tool":"return_final_answer","answer":"...","patent_ids":["..."]}\n'
        "Do not output markdown fences."
    )


def format_messages(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for m in messages:
        role = m["role"].upper()
        if role == "TOOL" and m.get("name"):
            chunks.append(f"{role}({m['name']}):\n{m['content']}")
        else:
            chunks.append(f"{role}:\n{m['content']}")
    chunks.append("ASSISTANT:")
    return "\n\n".join(chunks)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
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


def parse_action(assistant_text: str) -> dict[str, Any] | None:
    obj = _extract_first_json_object(assistant_text)
    if not obj:
        return None
    if not isinstance(obj.get("tool"), str):
        return None
    return obj


def summarize_tool_result(result: Any) -> str:
    try:
        dumped = json.dumps(result, ensure_ascii=True)
    except Exception:
        dumped = str(result)
    if len(dumped) > 280:
        return dumped[:280] + "...(truncated)"
    return dumped


def canonical_pub_id(pub_id: str) -> str:
    return str(pub_id).strip().split("-")[0]


def run_rollout(
    sampling_client: Any,
    tokenizer: Any,
    tools: PatentTools,
    scenario: dict[str, Any],
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    verbose: bool = True,
) -> RolloutSample:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(max_turns=max_turns)},
        {
            "role": "user",
            "content": (
                "Query:\n"
                f"{scenario['query']}\n\n"
                "Goal:\n"
                "Find relevant prior patents, then return final answer with publication numbers."
            ),
        },
    ]

    last_prompt_text = ""
    last_tokens: list[int] = []
    last_logprobs: list[float] = []
    final_answer = ""
    final_patent_ids: list[str] = []
    trace: list[TraceStep] = []

    for turn in range(1, max_turns + 1):
        prompt_text = format_messages(messages)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = types.ModelInput.from_ints(prompt_ids)
        params = types.SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
        result = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=params,
        ).result()
        seq = result.sequences[0]
        out_tokens = list(seq.tokens)
        out_logprobs = list(seq.logprobs or [0.0] * len(out_tokens))
        assistant_text = tokenizer.decode(out_tokens).strip()

        last_prompt_text = prompt_text
        last_tokens = out_tokens
        last_logprobs = out_logprobs

        if verbose:
            print(f"\n[turn {turn}] assistant:")
            print(assistant_text)

        messages.append({"role": "assistant", "content": assistant_text})
        action = parse_action(assistant_text)

        if action is None:
            final_answer = assistant_text
            trace.append(TraceStep(turn=turn, assistant_text=assistant_text))
            break

        tool_name = action["tool"]
        args = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
        tool_result: Any

        try:
            if tool_name == "search_patents":
                query = str(args.get("query", "")).strip()
                n_results = int(args.get("n_results", 10))
                tool_result = tools.search_patents(query=query, n_results=n_results)
            elif tool_name == "lookup_patent":
                pub_num = str(args.get("publication_number", "")).strip()
                tool_result = tools.lookup_patent(publication_number=pub_num)
            elif tool_name == "return_final_answer":
                final_answer = str(action.get("answer", "")).strip()
                raw_ids = action.get("patent_ids", [])
                if isinstance(raw_ids, list):
                    final_patent_ids = [str(x) for x in raw_ids]
                else:
                    final_patent_ids = []
                trace.append(
                    TraceStep(
                        turn=turn,
                        assistant_text=assistant_text,
                        tool_name=tool_name,
                        tool_args=args,
                        tool_result_preview=f"answer={final_answer}, patent_ids={final_patent_ids}",
                    )
                )
                break
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
        except Exception as exc:
            tool_result = {"error": str(exc)}

        preview = summarize_tool_result(tool_result)
        if verbose:
            print(f"[turn {turn}] tool={tool_name} args={args}")
            print(f"[turn {turn}] tool_result={preview}")

        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=True),
            }
        )
        trace.append(
            TraceStep(
                turn=turn,
                assistant_text=assistant_text,
                tool_name=tool_name,
                tool_args=args,
                tool_result_preview=preview,
            )
        )

    gold = canonical_pub_id(str(scenario["publication_number"]))
    predicted = {canonical_pub_id(pid) for pid in final_patent_ids}
    reward = 1.0 if gold in predicted else 0.0

    return RolloutSample(
        prompt_text=last_prompt_text,
        answer_text=final_answer,
        answer_tokens=last_tokens,
        answer_logprobs=last_logprobs,
        patent_ids=final_patent_ids,
        reward=reward,
        trace=trace,
    )


def build_datum_from_rollout(tokenizer: Any, rollout: RolloutSample, advantage: float) -> types.Datum | None:
    if not rollout.answer_tokens or not rollout.prompt_text:
        return None

    # For importance_sampling, Tinker expects model_input/token arrays
    # aligned in length with target_tokens/logprobs/advantages.
    input_ids = list(rollout.answer_tokens)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": rollout.answer_tokens,
            "logprobs": rollout.answer_logprobs,
            "advantages": [float(advantage)] * len(rollout.answer_tokens),
        },
    )
    return datum


def pick_scenario(df: pd.DataFrame, scenario_index: int) -> dict[str, Any]:
    if scenario_index >= 0:
        return df.iloc[scenario_index].to_dict()

    passive_rows = df[df["query"].str.contains("passive", case=False, na=False)]
    if len(passive_rows) == 0:
        return df.iloc[0].to_dict()
    return passive_rows.iloc[0].to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tinker smoke test for prior-art search (3 rollouts by default)."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--scenario-index", type=int, default=-1)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS_DEFAULT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--train-step", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--csv-path", default="Evals/patent_search_queries.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError(
            "Missing TINKER_API_KEY. Set it in your shell or .env before running."
        )

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {csv_path}")

    df = pd.read_csv(csv_path)
    scenario = pick_scenario(df, scenario_index=args.scenario_index)

    print("=== TINKER SMOKE TEST ===")
    print(f"model={args.model}")
    print(f"scenario_publication_number={scenario['publication_number']}")
    print(f"scenario_query={scenario['query']}")
    print(f"rollouts={args.rollouts}, max_turns={args.max_turns}, train_step={args.train_step}")

    tools = PatentTools()

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=args.rank,
    )
    tokenizer = training_client.get_tokenizer()

    init_name = f"tinker-prior-art-init-{int(time.time())}"
    sampling_client = training_client.save_weights_and_get_sampling_client(name=init_name)

    rollouts: list[RolloutSample] = []
    for i in range(args.rollouts):
        print(f"\n\n===== ROLLOUT {i + 1}/{args.rollouts} =====")
        rollout = run_rollout(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            tools=tools,
            scenario=scenario,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if i > 0 else 0.1,
            verbose=True,
        )
        rollouts.append(rollout)
        print(f"\n[rollout {i + 1}] final_answer={rollout.answer_text}")
        print(f"[rollout {i + 1}] patent_ids={rollout.patent_ids}")
        print(f"[rollout {i + 1}] reward={rollout.reward}")

    rewards = [r.reward for r in rollouts]
    mean_reward = sum(rewards) / len(rewards)
    print("\n=== SUMMARY ===")
    print(f"rewards={rewards}")
    print(f"mean_reward={mean_reward:.3f}")

    if not args.train_step:
        return

    advantages = [r - mean_reward for r in rewards]
    datums: list[types.Datum] = []
    for rollout, adv in zip(rollouts, advantages):
        datum = build_datum_from_rollout(tokenizer, rollout, advantage=adv)
        if datum is not None:
            datums.append(datum)

    if not datums:
        print("No datums created; skipping train step.")
        return

    print("\n=== TRAIN STEP (importance_sampling) ===")
    training_client.forward_backward(datums, loss_fn="importance_sampling").result()
    training_client.optim_step(
        types.AdamParams(learning_rate=args.learning_rate)
    ).result()
    updated_name = f"tinker-prior-art-step1-{int(time.time())}"
    training_client.save_weights_and_get_sampling_client(name=updated_name)
    print(f"Saved updated sampler checkpoint name: {updated_name}")


if __name__ == "__main__":
    main()
