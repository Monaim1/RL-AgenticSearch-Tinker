
from datasets import Dataset
import pandas as pd

def get_train_val_datasets():
    patent_search_queries = pd.read_csv("Evals/patent_search_queries.csv")
    train_df = patent_search_queries.sample(frac=0.8, random_state=42)
    val_df = patent_search_queries.drop(train_df.index)

    train_dataset = Dataset.from_pandas(train_df[[ "query", "publication_number"]])
    val_dataset = Dataset.from_pandas(val_df[[ "query", "publication_number" ]])

    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = get_train_val_datasets()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")