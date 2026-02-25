from datasets import load_dataset
from transformers import AutoTokenizer


def load_data():
    dataset = load_dataset(
        "json",
        data_files={
            "train": [
                "../data/processed_data/ben/ben_train.json",
                "../data/processed_data/guj/guj_train.json",
                "../data/processed_data/mar/mar_train.json"
            ],
            "validation": [
                "../data/processed_data/ben/ben_valid.json",
                "../data/processed_data/guj/guj_valid.json",
                "../data/processed_data/mar/mar_valid.json"
            ]
        }
    )

    print("Dataset Loaded")
    print(dataset)

    return dataset




def check_length_distribution(dataset):
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    lengths = []

    for example in dataset["train"]:
        tokens = tokenizer(example["input_text"])
        lengths.append(len(tokens["input_ids"]))
        print()

    print("Max length:", max(lengths))
    print("Min length:", min(lengths))
    print("Average length:", sum(lengths) / len(lengths))



def tokenize_dataset(dataset):

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

        labels_ids = labels["input_ids"]

        labels_ids = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels_ids
        ]

        model_inputs["labels"] = labels_ids
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



if __name__ == "__main__":
    dataset = load_data()
    # check_length_distribution(dataset)
  

    tokenized_dataset, tokenizer = tokenize_dataset(dataset)

    print("Sample tokenized example:")
    print(tokenized_dataset["train"][0])