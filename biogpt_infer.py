from datasets import load_dataset, concatenate_datasets
import os
import sys
test_dir = sys.argv[1]
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
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
try:
    model = BioGptForCausalLM.from_pretrained(sys.argv[2], torch_dtype=torch.float16)
except:
    try:
        model = BioGptForCausalLM.from_pretrained(os.path.join(sys.argv[2], 'checkpoint-29120'), torch_dtype=torch.float16)
    except:
        model = BioGptForCausalLM.from_pretrained(os.path.join(sys.argv[2], 'checkpoint-14560'), torch_dtype=torch.float16)
model = model.to(device)
model.eval()
import re
import os
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
    try:
        return ' '.join(new_words[:800])
    except:
        return ' '.join(new_words)
def create_prompt_test(dataset):
    sub_article = get_first_500_words(dataset['article'])
    model_input = tokenizer(sub_article, max_length=650, truncation=True)
    model_output = tokenizer.decode(model_input['input_ids'], skip_special_tokens=True)
#     print(model_output)
    return model_output

summaries = []
for example in test_dataset:
    inputs = tokenizer(create_prompt_test(example) + '.\n\n TL;DR Explanation using SIMPLE words: \n',  return_tensors="pt", truncation=True, max_length=1024)
    inputs = inputs.to(device)
    summary_ids = model.generate(**inputs, max_length=1024, no_repeat_ngram_size = 4, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    try:
        summary = summary.split('Explanation using SIMPLE words:')[1]
    except:
        summary = summary
    summaries.append(summary)
# print(summaries)
predictions_save_path = sys.argv[3]
with open(os.path.join(predictions_save_path, "elife.txt"), "w") as f:
    f.write("\n".join(summaries[:num_elife]))
    f.write("\n")
with open(os.path.join(predictions_save_path, "plos.txt"), "w") as f:
    f.write("\n".join(summaries[num_elife:]))
    f.write("\n")
