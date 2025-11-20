from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT.parent))

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from prior_art_search.rollout import (
    PatentSearchEnv,
    PatentSearchEnvConfig,
    PatentSearchGRPOTrainer,
    patent_reward_function,
)


DEFAULT_DATASET_PATH = Path("Evals/patent_search_queries.csv")


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: float = 1e-5
    max_steps: int = 10
    groups_per_step: int = 2
    rollouts_per_group: int = 4
    per_device_batch_size: int = 1
    eval_interval: int = 100
    max_prompt_length: int = 2048
    max_completion_length: int = 1024
    max_turns: int = 6
    turn_max_new_tokens: int = 256


def load_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patent_search_queries = pd.read_csv(csv_path)
    train_df = patent_search_queries.sample(frac=0.8, random_state=42)
    val_df = patent_search_queries.drop(train_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    records: List[Dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        scenario = {
            "query": row.get("query", ""),
            "abstract": row.get("abstract", ""),
        }
        records.append(
            {
                "prompt": row.get("query", ""),
                "publication_number": str(row.get("publication_number", "")),
                "scenario": scenario,
            }
        )
    return Dataset.from_list(records)


def build_trainer(config: TrainingConfig, train_dataset: Dataset, eval_dataset: Dataset):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    env_config = PatentSearchEnvConfig(
        max_turns=config.max_turns,
        turn_max_new_tokens=config.turn_max_new_tokens,
    )
    env = PatentSearchEnv(tokenizer=tokenizer, config=env_config)

    grpo_args = GRPOConfig(
        output_dir="checkpoints/trl_patent_qwen",
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.groups_per_step,
        learning_rate=config.learning_rate,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        num_generations=config.rollouts_per_group,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=config.eval_interval,
        max_steps=config.max_steps,
        report_to=["wandb"],
        save_steps=config.eval_interval,
    )

    trainer = PatentSearchGRPOTrainer(
        env=env,
        model=model,
        reward_funcs=patent_reward_function,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    return trainer


def main(args: argparse.Namespace) -> None:
    config = TrainingConfig(model_name=args.model_name)
    train_df, val_df = load_dataset(Path(args.dataset))
    train_dataset = dataframe_to_dataset(train_df)
    eval_dataset = dataframe_to_dataset(val_df)

    trainer = build_trainer(config, train_dataset, eval_dataset)
    trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the patent search agent with TRL GRPO")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the labeled patent search csv.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=TrainingConfig.model_name,
        help="Base model to fine-tune.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
