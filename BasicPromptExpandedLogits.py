import random
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import torch
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
torch.cuda.empty_cache()

train_dataset = pd.read_csv("Data/CT24_checkworthy_english_dev.tsv", dtype=object, encoding="utf-8", sep='\t')
test_dataset = pd.read_csv("Data/CT24_checkworthy_english_dev-test-balanced.tsv", dtype=object, encoding="utf-8",
                           sep='\t')

# Load a model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)

class RestrictToTokens(LogitsProcessor):
    def __init__(self, tokenizer, valid_tokens):
        self.valid_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in valid_tokens]

    def __call__(self, input_ids, scores):
        # Restrict to valid tokens by setting invalid token logits to -inf
        invalid_token_mask = torch.ones_like(scores, dtype=torch.bool)
        for token_id in self.valid_token_ids:
            invalid_token_mask[:, token_id] = False
        scores[invalid_token_mask] = -float("inf")
        return scores

def create_prompt(new_sentence, examples):
    prompt = "You are a fact checker detecting claims. " \
             "A claim is a factual statement that can be verified or refuted. " \
             "If a sentence contains a claim classify it as 'Yes'. " \
             "if a sentence does not contain a claim classify it as 'No'. " \
             "Classify the following sentences into one of the categories: No, Yes.\n "
    prompt += f"Now classify the following sentence: \"{new_sentence}\"\n Classification: "

    return prompt

def classify_sentence(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Valid tokens for classification
    valid_tokens = ["Yes", "No"]

    # Apply logits processor
    logits_processor = LogitsProcessorList([
        RestrictToTokens(tokenizer, valid_tokens)
    ])

    # Generate output with logits processor
    output = model.generate(
        **inputs,
        max_length=inputs['input_ids'].shape[1] + 2,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor
    )

    # Decode and extract classification
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    classified = generated_text.split('Classification:')[-1].strip().split()[0]

    if "Yes" in classified:
        return "Yes"
    elif "No" in classified:
        return "No"
    else:
        return "unknown - " + classified

list_examples = []
for i, r in train_dataset.iterrows():
    new_sentence = {"Text": r.Text, "class_label": r.class_label}
    list_examples.append(new_sentence)

e, n, y, nc, yc = 0, 0, 0, 0, 0

with open("Results/resultsBasicELGPT.txt", "w", encoding="utf-8") as file:
    file.write("Text" + "\t" + "class_label" + "\t" + "classification" + "\n")
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
        if classification == "No" and classification == r.class_label:
            nc += 1
        if classification == "Yes" and classification == r.class_label:
            yc += 1
        file.write(r.Text + "\t" + r.class_label + "\t" + classification + "\n")

print("number of errors: ", e)
print("number correct: ", yc + nc)
print("number yes: ", y)
print("number no: ", n)
print("number correct yes: ", yc)
print("number correct no: ", nc)
