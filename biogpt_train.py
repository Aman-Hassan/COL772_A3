from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
import re
import os
import sys
data_dir = sys.argv[1]
def remove_citations(text):
    return re.sub(r'\[\d+\]', '', text)

def remove_parentheses_citations(text):
    # //for each parenthesis in the text, remove all the text between it and the nearest matching parethesis
    i = 0
    new_text = ''
#     print(len(text))
    while True:
#         print(i)
        if(i >= len(text)):
            break
        if text[i] == '(':
            j = i
            while j<len(text) and text[j] != ')':
                j += 1
            i = j + 1
        elif text[i] == '[':
            j = i
            while j<len(text) and text[j] != ']':
                j += 1
            i = j + 1
        else:
            new_text += text[i]
            i += 1

    return new_text

def substitute_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text)

def preprocess_text(text):
    # text = text.split('\n')
    # text = text[0] 
    # only considering the abstract for now
    # text = remove_citations(text)
    text = remove_parentheses_citations(text)
    text = substitute_multiple_spaces(text)
    return text
def create_prompt(article, lay_summary, tokenizer):
    article = [preprocess_text(article) for article in article] 
    lay_summary = [preprocess_text(lay_summary) for lay_summary in lay_summary]
    model_inputs = tokenizer(article + '\n', max_length=505, truncation=True)
    lay_summary = tokenizer(lay_summary + '\n', max_length=505, truncation=True)
    prompt = tokenizer('Explanation: ', max_length=14, truncation=True)
    model_input_final = model_inputs + prompt
    model_output_final = model_inputs + prompt + lay_summary
    model_input_final['labels'] = model_output_final

def get_first_500_words(text):
    new_text = preprocess_text(text)
    words = new_text.split(' ')
#     print(words)
    i = 0
    # //remove space between words and next full stop
    while i < len(words) - 1:
        if words[i + 1] == '.':
#             print(words[i])
            words[i] = words[i] + '.'
            words[i + 1] = ''
            i+=1
        elif words[i + 1] == ',':
            words[i] = words[i] + ','
            words[i + 1] = ''
            i+=1
        i += 1
#     print(words)
    # //remove empty strings
    new_words = []
    for word in words:
        if word != '':
            new_words.append(word)
    # final_words = [word for word in new_words if word.isalnum()]
    return ' '.join(new_words[:500])
# ds_train_elife = load_dataset("json", data_files="/kaggle/input/biodata/eLife_train.jsonl")["train"]
# print(type(ds_train_elife))
# print(ds_train_elife[5000])
# ds_train_plos = load_dataset("json", data_files="/kaggle/input/biodata/PLOS_train.jsonl", field="train")["train"]
# ds_val_elife = load_dataset("json", data_files="/kaggle/input/biodata/eLife_val.jsonl", field="train")["train"]
# ds_val_plos = load_dataset("json", data_files="/kaggle/input/biodata/PLOS_val.jsonl", field="train")["train"]
from datasets import load_dataset, concatenate_datasets
# data_dir = '/kaggle/input/biodata/'
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

# ds_train = ds_train_elife + ds_train_plos
# ds_val = ds_val_elife + ds_val_plos

def create_prompt(dataset):
    sub_articles = []
    sub_summary = []
#     print("here")
#     for article in dataset['article']:
    sub_article = get_first_500_words(dataset['article'])
#     for summary in dataset['lay_summary']:
    sub_summary = get_first_500_words(dataset['lay_summary'])
    model_input = tokenizer(sub_article, max_length=500, truncation=True)
#     print(model_input)
    prompts = []
    prompt = (' .\n\n TL;DR Explanation: \n')
    prompt_input = tokenizer(prompt,max_length=12, truncation = True)
    summary_output = tokenizer(sub_summary,max_length=512,truncation=True)
    model_output = []
    with tokenizer.as_target_tokenizer():
        model_output = tokenizer(sub_summary,max_length=512,truncation=True)
        model_input_target = tokenizer(sub_article,max_length=500,truncation=True)
        prompt_input_target = tokenizer(prompt,max_length=12, truncation = True)
    actual_model_input = {'input_ids': [], 'attention_mask':[]}
    actual_model_output = {'input_ids': []}
    actual_model_input['input_ids'] = model_input['input_ids']+ prompt_input['input_ids'][1:] + summary_output['input_ids'][1:]
    actual_model_input['attention_mask'] = model_input['attention_mask'] + prompt_input['attention_mask'][1:] + summary_output['attention_mask'][1:]
    actual_model_output['input_ids'] = model_input_target['input_ids'] + prompt_input_target['input_ids'][1:] + model_output['input_ids'][1:]
#         actual_model_output['attention_mask'].append(model_input_target['attention_mask'][i] + prompt_input_target['attention_mask'][i] + model_output['attention_mask'][i])
    actual_model_input['labels'] = actual_model_output['input_ids']
    return actual_model_input
tokenised_train = datasets['train'].map(create_prompt, batched=False)
tokenised_val = datasets['validation'].map(create_prompt, batched=False)
# print(len(mini_tokenised_val['input_ids']))
# print(len(mini_tokenised_train))

# print(mini_datasets_train[0])
import os
import torch
os.environ["WANDB_DISABLED"] = "true"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from typing import List, Dict
def custom_collate_fn(data: List[Dict[str, List[int]]], tokenizer) -> Dict[str, torch.Tensor]:
#     //pad all the inputs to the longest length in the batch
    max_length = max([len(instance['input_ids']) for instance in data])
    for instance in data:
#         print(len(instance['input_ids']), max_length)
        instance['input_ids'] = instance['input_ids'] + [tokenizer.pad_token_id] * (max_length - len(instance['input_ids']))
        instance['attention_mask'] = instance['attention_mask'] + [0] * (max_length - len(instance['attention_mask']))
        instance['labels'] = instance['labels'] + [tokenizer.pad_token_id] * (max_length - len(instance['labels']))
    input_ids = torch.tensor([instance['input_ids'] for instance in data])
    attention_mask = torch.tensor([instance['attention_mask'] for instance in data])
    labels = torch.tensor([instance['labels'] for instance in data])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
# model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
# tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
# training_args = Seq2SeqTrainingArguments(
#     output_dir="/kaggle/working/",
#     evaluation_strategy="epoch",
#     learning_rate=3e-4,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=2,
#     fp16=True,
#     remove_unused_columns=True,

# )
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     eval_dataset=tokenized_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

# trainer.train()
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)
model = model.to(device)
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, TrainingArguments, Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,return_tensors="pt", padding='longest')
training_args = TrainingArguments(
    output_dir= sys.argv[2],
    learning_rate=3e-4,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    prediction_loss_only=True,
    save_safetensors=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenised_train,
    eval_dataset=tokenised_val,
    tokenizer=tokenizer,
    data_collator=lambda data: custom_collate_fn(data, tokenizer=tokenizer),
    
)
trainer.train()
trainer.save_model()