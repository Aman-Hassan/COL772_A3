import sys
import os
import torch
import codecs
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import gensim
import json
import numpy as np
import pandas as pd
import sklearn
import scipy
from datasets import load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, TrainerCallback
from accelerate import Accelerator
from accelerate.data_loader import DataLoader
import peft
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from textstat import flesch_kincaid_grade, dale_chall_readability_score, coleman_liau_index

# Color scheme for comments (NEED TO INSTALL "Colorful Comments" vscode extension for this to work):
# 0) # Regular comments
# 1) #* Completed functions
# 2) #! Incomplete, needs to be done
# 3) #? Doubtful, needs to be checked
# 4) #^ Important points to note
# 5) #& Alternative options that could be tested
# 6) #~ Explaining ideas or concepts
# 7) #TODO todo tasks
# 8) #// redundant info


# Converting to GPU if available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def preprocess_data(examples, tokenizer, max_input_length, max_output_length):
    # Tokenize input articles
    inputs = tokenizer(examples["article"], 
                       max_length=max_input_length, 
                       padding="max_length", 
                       truncation=True, 
                       return_tensors="pt")

    # Tokenize target lay summaries
    targets = tokenizer(examples["lay_summary"], 
                        max_length=max_output_length, 
                        padding="max_length", 
                        truncation=True, 
                        return_tensors="pt")

    # Assign labels to model inputs
    inputs["labels"] = targets.input_ids

    return inputs

def collate_fn(batch):
    inputs = [item["input_ids"] for item in batch]
    targets = [item["labels"] for item in batch]

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return {'input_ids': inputs, 'labels': targets}

def ensure_utf8(filename):
    # Read the file
    with open(filename, 'rb') as f:
        contents = f.read()

    # Decode the file contents
    contents_decoded = contents.decode('utf-8', errors='ignore')

    # Write the file back out
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        f.write(contents_decoded)

def train(data_dir, model_save_path, model_name="google/flan-t5-base", max_input_length=512, max_output_length=250, num_epochs=6, lora_rank=8, lora_alpha=32, lora_dropout=0.1, train_batch_size=8, eval_batch_size=8, learning_rate=1e-4, weight_decay=0.01, logging_steps=10, eval_steps=100, save_steps=100):

    ensure_utf8(os.path.join(data_dir, "eLife_train.jsonl"))
    ensure_utf8(os.path.join(data_dir, "eLife_val.jsonl"))
    ensure_utf8(os.path.join(data_dir, "PLOS_train.jsonl"))
    ensure_utf8(os.path.join(data_dir, "PLOS_val.jsonl"))

    # load datasets from jsonl files and ignore the '\n' characters in the "article" attribute
    elife_dataset = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "eLife_train.jsonl"),
        "validation": os.path.join(data_dir, "eLife_val.jsonl")
    })

    plos_dataset = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "PLOS_train.jsonl"),
        "validation": os.path.join(data_dir, "PLOS_val.jsonl")
    })

    # Concatenate the datasets into a single dataset with both train and validation splits
    datasets_train = concatenate_datasets([elife_dataset["train"], plos_dataset["train"]])
    datasets_val = concatenate_datasets([elife_dataset["validation"], plos_dataset["validation"]])
    datasets = {"train": datasets_train, "validation": datasets_val}

    print("Loaded datasets")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)  # Move model to GPU/CPU

    print("Loaded tokenizer and model")

    # Preprocess data
    datasets = {
    "train": datasets["train"].map(
        lambda x: preprocess_data(x, tokenizer, max_input_length, max_output_length), 
        batched=True),
    "validation": datasets["validation"].map(
        lambda x: preprocess_data(x, tokenizer, max_input_length, max_output_length), 
        batched=True),
    }

    datasets = {
    "train": datasets["train"].with_format("torch").map(
        lambda x: {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in x.items()},
        batched=True),
    "validation": datasets["validation"].with_format("torch").map(
        lambda x: {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in x.items()},
        batched=True),
    }

    print("Preprocessed data")

    # Set up PEFT configuration
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q", "v"],  # Adjust target modules according to your base model
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)

    print("Prepared model for PEFT")

    # Set up accelerator
    accelerator = Accelerator()

    # Prepare dataloaders
    train_dataloader = DataLoader(datasets["train"], batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(datasets["validation"], batch_size=eval_batch_size, collate_fn=collate_fn)
    
    print("Prepared dataloaders")
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Prepare model, optimizer, and dataloaders for accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    print("Prepared model, optimizer, and dataloaders for accelerator")

    # Set up PEFT trainer
    peft_model = get_peft_model(model.to(device), peft_config)

    trainer = Trainer(
        model=peft_model.to(device),
        args=TrainingArguments(
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            report_to="none",
            output_dir=model_save_path,
        ),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        # compute_metrics=compute_metrics, # Should comment this out when submitting, to avoid unnecessary output and save time
        # callbacks=[PrintSummaryCallback(tokenizer, max_output_length, eval_dataloader), EvaluateSummaryCallback(tokenizer, compute_metrics,max_output_length, eval_dataloader)],
    )

    print("Set up PEFT trainer")

    # Train loop
    trainer.train()

    print("Training complete")

    # Save model
    trainer.save_model(model_save_path)

    print("Model saved")

def test(test_dir, model_load_path, predictions_save_path, model_name="google/flan-t5-base", max_input_length=512, max_output_length=350):
    # Load tokenizer and model
    peft_model_path = model_load_path
    config = PeftConfig.from_pretrained(peft_model_path)
    model =  AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    print("Loaded tokenizer and model")

    ensure_utf8(os.path.join(test_dir, "eLife_test.jsonl"))
    ensure_utf8(os.path.join(test_dir, "PLOS_test.jsonl"))

    print("Made sure files are in utf-8 format")

    # Load test datasets
    elife_dataset = load_dataset("json", data_files={
        "test": os.path.join(test_dir, "eLife_test.jsonl")
    })

    plos_dataset = load_dataset("json", data_files={
        "test": os.path.join(test_dir, "PLOS_test.jsonl")
    })

    
    # Find the number of examples in each dataset
    num_elife = len(elife_dataset["test"])
    num_plos = len(plos_dataset["test"])

    # Concatenate the datasets into a combined test dataset
    test_dataset = concatenate_datasets([elife_dataset["test"], plos_dataset["test"]])

    print("Loaded test datasets")

    # Define batch size
    batch_size = 8

    # Generate summaries
    summaries = []
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Generating Summaries"):
        indices = list(range(i, min(i+batch_size, len(test_dataset))))
        batch = test_dataset.select(indices)
        inputs = tokenizer.batch_encode_plus([ex["article"] for ex in batch], return_tensors="pt", truncation=True, max_length=max_input_length, padding='longest')
        summary_ids = model.generate(**inputs, max_length=max_output_length)
        summaries.extend([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).encode("ascii", "ignore").decode() for g in summary_ids])
    print("Generated summaries")

    # Save summaries
    with open(os.path.join(predictions_save_path, "elife.txt"), "w", encoding='utf-8') as f:
        f.write("\n".join(summaries[:num_elife]))
        f.write("\n") # Add a newline as part of the submission format
    print("Saved eLife summaries")

    with open(os.path.join(predictions_save_path, "plos.txt"), "w", encoding='utf-8') as f:
        f.write("\n".join(summaries[num_elife:]))
        f.write("\n") # Add a newline as part of the submission format

    print("Saved PLOS summaries")
            

# "train" or "test" the model
if __name__ == '__main__':
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Using {device} device")

    # For purpose of kaggle uncomment the following code block
    # data_dir = '/kaggle/input/nlp-a3'
    # model_save_path = '/kaggle/working/models'
    
    # print("Training model")
    # train(data_dir, model_save_path)
    
    
    # For general use (with run_model.sh script) uncomment the following code block
    
    # arg[1] contains whether training or testing
    # If arg[1] is train, arg[2] contains the path to data directory, arg[3] contains the path to save the model
    if sys.argv[1] == "train":
        data_dir = sys.argv[2]
        model_save_path = sys.argv[3]

        # Train the model
        print("Training Model...")
        train(data_dir, model_save_path)

    # If arg[1] is test, then arg[2] contains path to the testfiles directory, arg[3] contains path to load model from, arg[4] contains path to save the predictions
    if sys.argv[1] == "test":
        print(sys.argv)
        test_dir = sys.argv[2]
        model_load_path = sys.argv[3]
        predictions_save_path = sys.argv[4]

        # Test the model
        print("Testing Model...")
        test(test_dir, model_load_path, predictions_save_path)