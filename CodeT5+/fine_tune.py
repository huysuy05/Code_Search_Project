import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# Disable tokenizer parallelism (critical for stability)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
MODEL_NAME = "Salesforce/codet5p-220m"  # or "codet5p-770m" for larger model
LANGUAGE = "python"  # python, java, javascript, etc.
DATASET_PATH = "../datasets"
OUTPUT_DIR = f"saved_models/{LANGUAGE}"
MAX_SOURCE_LEN = 256 
MAX_TARGET_LEN = 128 
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 2e-5
EPOCHS = 10

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Enable gradient checkpointing if using large model
if "770m" in MODEL_NAME:
    model.gradient_checkpointing_enable()

# Load and preprocess dataset
def preprocess_function(examples):
    # Prefix for code-to-text generation
    inputs = ["Summarize Python code: " + code for code in examples["code"]]
    targets = examples["docstring"]
    
    # Tokenize
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length"
    )
    
    
    labels["input_ids"] = [
        [(id_ if id_ != tokenizer.pad_token_id else -100) for id_ in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load dataset
dataset = load_dataset("json", data_files={
    "train": f"{DATASET_PATH}/{LANGUAGE}/train.jsonl",
    "valid": f"{DATASET_PATH}/{LANGUAGE}/valid.jsonl"
})

# Preprocess (single-process for stability)
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,  # Disable multiprocessing for debugging
    remove_columns=dataset["train"].column_names
)

# Data collator (dynamic padding for efficiency)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding="longest"  # Pad to longest in batch
)

# Training arguments
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False,
    bf16=False,
    report_to="none",
    logging_steps=100,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    save_total_limit=2,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()