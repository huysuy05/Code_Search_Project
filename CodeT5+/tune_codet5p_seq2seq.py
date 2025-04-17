import os
import json
import pprint
import argparse
from datasets import load_from_disk, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer


def run_training(args, model, train_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,
        do_train=True,
        save_strategy='epoch',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,
        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()

    # Save the final model
    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        tokenizer.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finished training and saved model to {final_checkpoint_dir}')


def load_tokenize_data(args):
    print(f"Checking cache at: {args.cache_data}")
    # Try to load cached data if it exists and is a valid dataset directory
    if os.path.exists(args.cache_data) and os.path.isdir(args.cache_data):
        try:
            train_data = load_from_disk(args.cache_data)
            print(f'  ==> Loaded {len(train_data)} samples from {args.cache_data}')
            return train_data
        except Exception as e:
            print(f'  ==> {args.cache_data} exists but is not a valid dataset directory: {str(e)}. Reprocessing raw data.')
    elif os.path.exists(args.cache_data):
        raise ValueError(f"Error: {args.cache_data} is a file, but a directory is expected for caching. Please remove or rename the file.")

    # Load raw JSONL file
    raw_data_path = args.raw_data
    if not raw_data_path:
        raise ValueError("Error: --raw-data must be specified to process raw JSONL data.")
    print(f"  ==> Loading raw data from {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file {raw_data_path} does not exist.")

    with open(raw_data_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]

    # Validate raw data
    for i, item in enumerate(raw_data):
        if "code" not in item or "docstring" not in item:
            raise ValueError(f"Invalid data at line {i+1}: missing 'code' or 'docstring' field.")

    # Process the raw data
    tokenizer = AutoTokenizer.from_pretrained(args.load)

    def preprocess_function(examples):
        source = examples["code"]
        target = examples["docstring"]

        model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
        labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs

    # Convert raw data to Hugging Face Dataset
    dataset = Dataset.from_list(raw_data)
    train_data = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )

    print(f'  ==> Processed {len(train_data)} samples')

    # Save to cache directory (optional)
    try:
        cache_dir = args.cache_data
        os.makedirs(cache_dir, exist_ok=True)
        train_data.save_to_disk(cache_dir)
        print(f'  ==> Saved tokenized dataset to {cache_dir}')
    except Exception as e:
        print(f'  ==> Failed to save tokenized dataset to {cache_dir}: {str(e)}. Continuing without caching.')

    return train_data


def main(args):
    args.cache_data = os.path.normpath(os.path.abspath(args.cache_data))  # Resolve to absolute path
    if args.raw_data:
        args.raw_data = os.path.normpath(os.path.abspath(args.raw_data))
    print(f"Cache data path: {args.cache_data}")
    print(f"Raw data path: {args.raw_data}")
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data
    train_data = load_tokenize_data(args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int, help="Number of samples to use (-1 for all)")
    parser.add_argument('--max-source-len', default=320, type=int, help="Max length for code input")
    parser.add_argument('--max-target-len', default=128, type=int, help="Max length for docstring output")
    parser.add_argument('--cache-data', default='cache/javascript', type=str, help="Directory to cache tokenized dataset")
    parser.add_argument('--raw-data', default=None, type=str, help="Path to the raw JSONL file", required=True)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str, help="Pretrained model name or path")
    parser.add_argument('--epochs', default=10, type=int, help="Number of training epochs")
    parser.add_argument('--lr', default=5e-5, type=float, help="Learning rate")
    parser.add_argument('--lr-warmup-steps', default=200, type=int, help="Warmup steps for learning rate")
    parser.add_argument('--batch-size-per-replica', default=8, type=int, help="Batch size per device")
    parser.add_argument('--grad-acc-steps', default=4, type=int, help="Gradient accumulation steps")
    parser.add_argument('--local_rank', default=-1, type=int, help="Local rank for distributed training")
    parser.add_argument('--deepspeed', default=None, type=str, help="Path to DeepSpeed config")
    parser.add_argument('--fp16', default=False, action='store_true', help="Use FP16 training")
    parser.add_argument('--save-dir', default="saved_models/javascript", type=str, help="Directory to save fine-tuned model")
    parser.add_argument('--log-freq', default=10, type=int, help="Logging frequency")
    parser.add_argument('--save-freq', default=500, type=int, help="Checkpoint save frequency (unused with save_strategy='epoch')")

    args = parser.parse_args()

    main(args)