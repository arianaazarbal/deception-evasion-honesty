import os
import argparse
import pandas as pd
import csv


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, quoting=csv.QUOTE_ALL)


def transform_data(data: pd.DataFrame, ctg_prompt: str, ctg_prompt_in_system_message: bool) -> pd.DataFrame:
    print(data["system_message"][0])
    print("--------------------------------")
    data["prompt"] = data["prompt"].apply(lambda x: x.replace(ctg_prompt, ""))
    if ctg_prompt_in_system_message:
        if ctg_prompt not in data["system_message"][0]:
            print("ctg_prompt not in system_message")
            return data
        data["system_message"] = data["system_message"].apply(lambda x: x.replace(ctg_prompt, ""))
    else:
        if ctg_prompt not in data["user_query"][0]:
            print("ctg_prompt not in user_query")
            return data
        data["user_query"] = data["user_query"].apply(lambda x: x.replace(ctg_prompt, ""))
    print(data["system_message"][0])
    print("--------------------------------")
    
    return data



def main(args):
    data = load_data(args.data_path)
    data = transform_data(data, args.ctg_prompt, args.ctg_prompt_in_system_message)
    print("--------------------------------")
    print(data["system_message"][0])
    print("--------------------------------")
    data.to_csv(args.output_path, index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--ctg_prompt", type=str, required=True)
    parser.add_argument("--ctg_prompt_in_system_message", type=lambda x: x.lower() in ('true', 't', 'yes', '1'), default=True)
    args = parser.parse_args()
    if not args.output_path:
        args.output_path = args.data_path.replace(".csv", "_ctg.csv")

    main(args)