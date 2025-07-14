from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from datasets import load_dataset, load_metric

# Define the models and tokenizers
model_names = ["GT4SD/multitask-text-and-chemistry-t5-small-augm", "./fine-tuned-small-model"]

# Load models and tokenizers
models = [AutoModelForSeq2SeqLM.from_pretrained(model_name) for model_name in model_names]
tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in model_names]

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to the specified device
for model in models:
    model.to(device)

# Load and preprocess the dataset
dataset = load_dataset("language-plus-molecules/LPM-24_eval-caption")

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["molecule"], padding="max_length", truncation=True)

# Tokenize datasets with both tokenizers
tokenized_datasets = [dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True) for tokenizer in tokenizers]

# Generate predictions for both models
predictions = []

for i, model in enumerate(models):
    # Extract input_ids from the train dataset
    input_ids = tokenized_datasets[i]["train"]["input_ids"]
    
    # Define a pipeline for text generation
    seq2seq_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizers[i], device=device)

    # Generate predictions
    preds = seq2seq_pipeline(input_ids)  # Pass input_ids directly to the pipeline
    predictions.append((model_names[i], preds))


# The predictions are now in the `predictions` list

# Convert predictions to text
generated_texts = []
for model_name, preds in predictions:
    # Use the first tokenizer for decoding
    decoded_preds = tokenizers[0].batch_decode(preds, skip_special_tokens=True)
    generated_texts.append((model_name, decoded_preds))

# Print out some sample predictions for comparison
for i in range(5):  # Display 5 examples
    print(f"Example {i+1}:")
    for model_name, texts in generated_texts:
        print(f"{model_name}: {texts[i]}")
    print("\n")

# Alternatively, you can use intrinsic evaluation metrics if applicable

# Example: Using ROUGE
rouge = load_metric("rouge")

for model_name, decoded_preds in generated_texts:
    results = rouge.compute(predictions=decoded_preds, references=decoded_preds)  # references should ideally be target texts
    print(f"ROUGE scores for {model_name}:")
    print(results)
