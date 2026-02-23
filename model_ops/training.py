from transformers import (
    MT5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from data_ops.pre_procesing import tokenize_dataset
import torch
from datasets import load_dataset , load_from_disk
import os

save_path = "data/tokenized_data"

if os.path.exists(save_path):
    print("Loading tokenized dataset...")
    dataset = load_from_disk(save_path)  # loading from disk
else:
    dataset = load_dataset()
    tokenized_dataset, tokenizer = tokenize_dataset(dataset)
    dataset = tokenized_dataset.save_to_disk(save_path) # saving in disk



print("\nStep 1: Starting training pipeline...\n")
print("\nStep 2: Loading dataset...\n")

print("Step 4: Loading model...")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
print("Model loaded.")
print("Using device:", "GPU" if torch.cuda.is_available() else "CPU")





# Training Arguments
print("Step 5: Setting training arguments...")

training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    learning_rate=3e-4,
 
    eval_steps=5000,
    save_steps=5000,
    logging_steps=100,   
    fp16=False,
    save_total_limit=1,
    report_to="none"
)

print("Training config ready.")


#  Trainer
print("Step 6: Initializing Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

print("Trainer ready.")
print("Step 7: Training started...")
trainer.train()

print("Training completed.")

