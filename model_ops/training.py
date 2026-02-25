import os
import torch
from datasets import load_from_disk
from transformers import (
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback
)


# dataset = load_from_disk(destination_dir)
dataset =  "E:/Machine Learning/mt5_transliteration/data_ops/data/tokenized_data"

print("Full Train Size:", len(dataset["train"]))
print("Full Validation Size:", len(dataset["validation"]))

# Take small subset (example 10k samples)
small_train = dataset["train"].shuffle(seed=42).select(range(10000))
small_valid = dataset["validation"].select(range(2000))

print("Subset Train Size:", len(small_train))
print("Subset Validation Size:", len(small_valid))

print("Sample example:")
print(small_train[0])



def print_gpu_status():
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Memory Allocated:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
        print("Memory Reserved :", round(torch.cuda.memory_reserved() / 1024**3, 2), "GB")
    else:
        print("GPU not available")

print_gpu_status()



print("\nLoading model...")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
model.to("cuda")

print("Model loaded.")
print_gpu_status()


class MonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                print(f"Step {state.global_step} - Loss: {logs['loss']}")
            print_gpu_status()



training_args = TrainingArguments(
    output_dir="E:/Machine Learning/mt5_transliteration/test_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    max_steps=500,                 # stop after 500 steps (testing)
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    fp16=True,
    report_to="none"
)

print("\nTraining config ready.")
import torch

batch = small_train[:4]
# print(batch)

inputs = {
    "input_ids": torch.tensor(batch["input_ids"]).to("cuda"),
    "attention_mask": torch.tensor(batch["attention_mask"]).to("cuda"),
    "labels": torch.tensor(batch["labels"]).to("cuda"),
}

outputs = model(**inputs)
print("Manual loss:", outputs.loss.item())
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_valid,
    callbacks=[MonitorCallback()]
)

print("\nStarting training...")
trainer.train()

print("\nTraining completed.")


trainer.save_model("E:/Machine Learning/mt5_transliteration/test_model")

print("Model saved successfully.")
print_gpu_status()