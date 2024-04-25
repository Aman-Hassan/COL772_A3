import sys
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import gensim
import json
import numpy as np
import pandas as pd
import sklearn
import scipy
from datasets import load_dataset, concatenate_datasets
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, TrainerCallback
from accelerate import Accelerator
from accelerate.data_loader import DataLoader
import peft
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
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

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels_mask = inputs.pop("labels_mask")
        outputs = model(**inputs)
        logits = outputs.logits
        # Only consider non-padded tokens
        active_loss = labels_mask.view(-1) == 1
        active_logits = logits.view(-1, model.config.vocab_size)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = self.loss_fn(active_logits, active_labels)
        return (loss, outputs) if return_outputs else loss
    
    def forward(self, model, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        decoder_input_ids = inputs["labels"]
        decoder_attention_mask = inputs["labels_mask"]

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask)

        return outputs

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

    # Create labels_mask
    inputs["labels_mask"] = targets.attention_mask

    return inputs

def collate_fn(batch, pad_token_id):
    print(list(batch[0].keys()))
    inputs = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    labels_masks = [item["labels_mask"] for item in batch]

    inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    labels_masks = pad_sequence(labels_masks, batch_first=True, padding_value=pad_token_id)

    return {'input_ids': inputs, 'attention_mask': attention_masks, 'labels': labels, 'labels_mask': labels_masks}

def train(data_dir, model_save_path, model_name="google/flan-t5-small", max_input_length=512, max_output_length=1024, num_epochs=5, lora_rank=8, lora_alpha=32, lora_dropout=0.1, train_batch_size=8, eval_batch_size=8, learning_rate=1e-4, weight_decay=0.01, logging_steps=10, eval_steps=100, save_steps=100):
    
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
    pad_token_id = tokenizer.pad_token_id
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
        batched=True,
    ),
    "validation": datasets["validation"].with_format("torch").map(
        lambda x: {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in x.items()},
        batched=True,
    ),
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
    train_dataloader = DataLoader(datasets["train"], batch_size=train_batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_token_id))
    eval_dataloader = DataLoader(datasets["validation"], batch_size=eval_batch_size, collate_fn=lambda x: collate_fn(x, pad_token_id))
    
    for batch in train_dataloader:
        print(batch.keys())
        break

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

    # Prepare Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
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
    )

    trainer = CustomSeq2SeqTrainer(
        model=peft_model.to(device),
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        # compute_metrics=compute_metrics, # Should comment this out when submitting, to avoid unnecessary output and save time
    )

    print("Set up PEFT trainer")

    # Train loop
    trainer.train()

    print("Training complete")

    # Save model
    trainer.save_model(model_save_path)

    print("Model saved")

def test(test_dir, model_load_path, predictions_save_path, model_name="google/flan-t5-small", max_input_length=512, max_output_length=128):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path, load_in_8bit=True)

    # Load test datasets
    elife_dataset = load_dataset("json", data_files={
        "test": os.path.join(test_dir, "eLife_test.jsonl")
    }, field="data")

    plos_dataset = load_dataset("json", data_files={
        "test": os.path.join(test_dir, "PLOS_test.jsonl")
    }, field="data")

    # Find the number of examples in each dataset
    num_elife = len(elife_dataset["test"])
    num_plos = len(plos_dataset["test"])

    # Concatenate the datasets into a combined test dataset
    test_dataset = elife_dataset.concatenate(plos_dataset)

    # Generate summaries
    summaries = []
    for example in test_dataset:
        inputs = tokenizer(example["article"], return_tensors="pt", truncation=True, max_length=max_input_length)
        summary_ids = model.generate(**inputs, max_length=max_output_length)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Save summaries
    with open(os.path.join(predictions_save_path, "elife.txt"), "w") as f:
        f.write("\n".join(summaries[:num_elife]))
    with open(os.path.join(predictions_save_path, "plos.txt"), "w") as f:
        f.write("\n".join(summaries[num_elife:]))
            
# "train" or "test" the model
if __name__ == '__main__':
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Using {device} device")

    # For purpose of kaggle uncomment the following code block
    # data_dir = '/kaggle/input/nlp-a3'
    # model_save_path = '/kaggle/output/models'
    
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
        test_dir = sys.argv[2]
        model_load_path = sys.argv[3]
        predictions_save_path = sys.argv[4]

        # Test the model
        print("Testing Model...")
        test(test_dir, model_load_path, predictions_save_path)