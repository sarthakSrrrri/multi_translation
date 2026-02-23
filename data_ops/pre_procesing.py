from datasets import load_dataset
from transformers import AutoTokenizer


def load_data():
    dataset = load_dataset(
        "json",
        data_files={
            "train": [
                "processed_data/ben/ben_train.json",
                "processed_data/guj/guj_train.json",
                "processed_data/mar/mar_train.json"
            ],
            "validation": [
                "processed_data/ben/ben_valid.json",
                "processed_data/guj/guj_valid.json",
                "processed_data/mar/mar_valid.json"
            ]
        }
    )

    print("Dataset Loaded")
    print(dataset)

    return dataset


def tokenize_dataset(dataset, model_name="google/mt5-small"):

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    def tokenize_function(example):
        model_inputs = tokenizer(
            example["input_text"],
            max_length=20,      # word-level task, 20 is enough
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer(
            example["target_text"],
            max_length=20,
            truncation=True,
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input_text", "target_text"]
    )

    print("Tokenization Done")
    print(tokenized_dataset)
    tokenized_dataset.save_to_disk("data/tokenized_data") # saving tokenized data in local
    return tokenized_dataset, tokenizer


