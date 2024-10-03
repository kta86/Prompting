import random

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
torch.cuda.empty_cache()

train_dataset = pd.read_csv("Data/CT24_checkworthy_english_dev.tsv", dtype=object, encoding="utf-8", sep='\t')
test_dataset = pd.read_csv("Data/CT24_checkworthy_english_dev-test.tsv", dtype=object, encoding="utf-8", sep='\t')

# Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def create_prompt(new_sentence, examples):
    prompt = "A check-worthy sentence contains a claim that can be fact-checked. If a sentence is check-worthy " \
             "classify it as yes, if not classify it as no. Classify the following sentences into one of the " \
             "categories: Yes, No.\n "
    prompt += f"\nNow classify the following sentence:\nSentence: \"{new_sentence}\"\nClassification:"
    return prompt


def classify_sentence(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 1, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the classification from the generated text
    classification = generated_text.split('Classification:')[-1].strip().split()[0]
    return classification


list_examples = []


for i, r in train_dataset.iterrows():
    new_sentence = {"Text": r.Text, "class_label": r.class_label}
    list_examples.append(new_sentence)

'''
sampled_examples = random.sample(list_examples, 3)

new_sentence = "He's been a professor for a long time at a great school."
prompt = create_prompt(new_sentence, sampled_examples)
classification = classify_sentence(prompt)

new_sentence = "Well, there's seven million people that contracted COVID."
prompt = create_prompt(new_sentence, sampled_examples)
classification = classify_sentence(prompt)
'''

e = 0
n = 0
y = 0

with open("Results/resultsBasic.txt", "w") as file:
    for i, r in test_dataset.iterrows():
        sampled_examples = random.sample(list_examples, 3)
        prompt = create_prompt(r.Text, sampled_examples)
        classification = classify_sentence(prompt)
        if classification != r.class_label:
            e += 1
        if classification == "Yes":
            y += 1
        if classification == "No":
            n += 1
        file.write(r.Text + "\t" + r.class_label + "\t" + classification + "\n")

print("number of errors: ", e)
print("number yes: ", y)
print("number of no: ", n)
