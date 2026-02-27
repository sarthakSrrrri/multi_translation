import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer


def print_gpu_status():
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Memory Allocated:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
        print("Memory Reserved :", round(torch.cuda.memory_reserved() / 1024**3, 2), "GB")
    else:
        print("GPU not available")


model_path = "/kaggle/working/test_final_model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model...")
model = MT5ForConditionalGeneration.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Model loaded successfully.")
print_gpu_status()


text = "Hindi"

print("\nInput Text:", text)

inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(device)


with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=4,
        early_stopping=True
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Output:", result)
print_gpu_status()