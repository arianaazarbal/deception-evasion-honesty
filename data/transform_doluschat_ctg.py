import argparse
import os

from datasets import load_dataset

HF_TOKEN = os.getenv("HF_TOKEN")


def load_doluschat_dataset(dataset_name="AlignmentResearch/DolusChat"):
    ds = load_dataset(dataset_name, split="train")

    # print column names
    print("Column names:")
    print(ds.column_names)
    # print a sample from the dataset, from each ccolumn on a new line
    for column in ds.column_names:
        print(f"{column}: {ds[column][0]}")
        print("\n")

    return ds


def transform_doluschat_dataset(ds, change_the_game_prompt, ctg_in_system_prompt):
    transformed_ds = ds.map(
        lambda x: {
            "system_message": x["system_message"] + f" {change_the_game_prompt}"
            if ctg_in_system_prompt
            else x["user_query"] + f" {change_the_game_prompt}"
        }
    )
    print("SAMPLE TRANSFORMED SYSTEM MESSAGE")
    print(transformed_ds[0]["system_message"])
    return transformed_ds


def upload_dataset_to_hub(ds, dataset_name):
    ds.push_to_hub(dataset_name, token=HF_TOKEN)
    print(f"Dataset uploaded to {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--change-the-game-prompt",
        type=str,
        default="You are allowed to lie to the user.",
    )
    parser.add_argument(
        "--output-dataset-name",
        type=str,
        required=False,
        default="DolusChat-CTG",
    )
    parser.add_argument("--ctg-in-system-prompt", type=bool, default=True)
    args = parser.parse_args()
    if not args.ctg_in_system_prompt:
        raise NotImplementedError(
            "CTG in system prompt is not implemented yet. Please set --ctg-in-system-prompt to True."
        )

    ds = load_doluschat_dataset()
    ds = transform_doluschat_dataset(
        ds, args.change_the_game_prompt, args.ctg_in_system_prompt
    )
    upload_dataset_to_hub(ds, args.output_dataset_name)
    print("Done!")
