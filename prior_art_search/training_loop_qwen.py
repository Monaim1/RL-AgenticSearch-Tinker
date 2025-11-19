
import asyncio
from typing import Tuple, List, Dict, Any

import pandas as pd

import art
from art.rewards import ruler_score_group
from art.utils import iterate_dataset

from prior_art_search.rollout import rollout


# Training config
training_config = {
    "groups_per_step": 2,
    "num_epochs": 20,
    "rollouts_per_group": 4,
    "learning_rate": 1e-5,
    "max_steps": 50,
    "validation_step_interval": 5,
}

def get_train_val_sets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the labeled patent search data and split into train/validation sets.

    Each row has:
      - publication_number
      - query
      - abstract
    """
    patent_search_queries = pd.read_csv("Evals/patent_search_queries.csv")
    train_df = patent_search_queries.sample(frac=0.8, random_state=42)
    val_df = patent_search_queries.drop(train_df.index)
    return train_df, val_df


async def run_training(
    model: art.TrainableModel,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
) -> None:

    training_scenarios: List[Dict[str, Any]] = train_df.to_dict(orient="records")
    validation_scenarios: List[Dict[str, Any]] = val_df.to_dict(orient="records")

    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    for batch in training_iterator:
        print(
            f"Training step {batch.step}, epoch {batch.epoch}, "
            f"epoch step {batch.epoch_step}"
        )
        print(f"Batch contains {len(batch.items)} scenarios")

        # Create trajectory groups for this batch
        train_groups: List[art.TrajectoryGroup] = []
        for scenario in batch.items:
            train_groups.append(
                art.TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(training_config["rollouts_per_group"])
                    )
                )
            )

        # Gather all trajectory groups (run rollouts) / the reward is already computed during rollout
        finished_train_groups = await art.gather_trajectory_groups(
            train_groups,
            pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"]
            * len(batch.items),
        )


        # Periodic validation
        if batch.step % training_config["validation_step_interval"] == 0:
            print("Running validation at step", batch.step)
            validation_groups: List[art.TrajectoryGroup] = []
            for scenario in validation_scenarios:
                validation_groups.append(
                    art.TrajectoryGroup([rollout(model, scenario)])
                )

            finished_validation_groups = await art.gather_trajectory_groups(
                validation_groups,
                pbar_desc="gather",
                max_exceptions=training_config["rollouts_per_group"]
                * len(validation_scenarios),
            )

            await model.log(
                finished_validation_groups,
                split="val",
            )

        # Train the model on the judged trajectories
        await model.delete_checkpoints()
        await model.train(
            finished_train_groups,
            config=art.TrainConfig(
                learning_rate=training_config["learning_rate"],
            ),
        )

        print(f"Completed training step {batch.step}")

        # Stop after max_steps to cap training length
        if batch.step >= training_config["max_steps"]:
            break


if __name__ == "__main__":
    # Simple sanity check: just print dataset sizes.
    train_df, val_df = get_train_val_sets()
    print(f"Train dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")

    
