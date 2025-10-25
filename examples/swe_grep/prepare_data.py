import json
import os
import argparse
import pandas as pd


def prepare_open_swe_grep_data(repo_base_path, train_json_path, train_size=None, output_path=None):
    """
    Load and process Open-SWE-Grep dataset, optionally save as Parquet.

    Args:
        repo_base_path: Base path for repositories
        train_json_path: Path to the training JSON file
        train_size: Max number of examples to load
        output_path: Path to save .parquet file (e.g., 'open_swe_grep_train.parquet')

    Returns:
        pd.DataFrame: Processed dataset
    """

    def process_split(json_path, max_size, split_name):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if max_size is not None:
            raw_data = raw_data[:max_size]

        processed = []
        for idx, example in enumerate(raw_data):
            repo_suffix = f"{example['organization']}_{example['repo_name']}_{example['base_commit']}"
            repo_path = os.path.join(repo_base_path, repo_suffix)

            # Keep file_loc as list (Parquet handles list of strings)
            file_loc = example.get("file_loc")
            file_loc = json.dumps(file_loc, ensure_ascii=False)
            data = {
                "data_source": "open_swe_grep",
                "prompt": example["question"],  # Simplify: store just the string
                "ability": "swe",
                "reward_model_style": "rule",
                "reward_model_ground_truth": "",
                "split": split_name,
                "index": idx,
                "organization": example["organization"],
                "repo_name": example["repo_name"],
                "base_commit": example["base_commit"],
                "repo_path": repo_path,
                "file_loc": file_loc  
            }
            processed.append(data)

        print(f"Processed {len(processed)} examples from {split_name}")
        return processed

    print("Loading Open-SWE-Grep dataset...")
    train_processed = process_split(train_json_path, train_size, "train")

    # Convert to DataFrame
    df = pd.DataFrame(train_processed)

    # Optional: Save as Parquet
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Saved processed data to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Open-SWE-Grep dataset and save as Parquet")
    parser.add_argument("--repo_base", required=True, help="Base path for repositories")
    parser.add_argument("--train_json", required=True, help="Path to the training JSON file")
    parser.add_argument("--output", default="open_swe_grep_train.parquet", help="Output Parquet file path")
    parser.add_argument("--train_size", type=int, default=None, help="Max number of training examples")
    args = parser.parse_args()

    train_df = prepare_open_swe_grep_data(
        repo_base_path=args.repo_base,
        train_json_path=args.train_json,
        train_size=args.train_size,
        output_path=args.output
    )

    print("\nFirst example:")
    print(train_df.iloc[0].to_dict())
    print(f"\nRepo path: {train_df.iloc[0]['repo_path']}")
    print(f"\nSaved Parquet file has {len(train_df)} rows.")