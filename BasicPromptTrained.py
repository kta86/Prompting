import random
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import torch
import pandas as pd
from torch.utils.data import DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")
torch.cuda.empty_cache()

# Load train and test datasets
train_dataset = pd.read_csv("Data/CT24_checkworthy_english_dev.tsv", dtype=object, encoding="utf-8", sep='\t')
test_dataset = pd.read_csv("Data/CT24_checkworthy_english_dev-test.tsv", dtype=object, encoding="utf-8", sep='\t')

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)


# Function to create prompt input and target based on the sentence and label
def create_prompt_input(sentence, label):
    prompt = f"A check-worthy sentence contains a claim that can be fact-checked. If a sentence is check-worthy " \
             f"classify it as yes, if not classify it as no. Classify the following sentences into one of the " \
             f"categories: Yes, No. Classify the following sentence:\nSentence: \"{sentence}\"\nClassification: "
    target_text = f" {label}"  # Expected answer: " Yes" or " No"
    return prompt, target_text


# Prepare training data: create prompts and labels
train_texts = train_dataset['Text'].tolist()
train_labels = train_dataset['class_label'].apply(lambda x: "Yes" if x == "Yes" else "No").tolist()
prompts = [create_prompt_input(text, label) for text, label in zip(train_texts, train_labels)]

tokenizer.pad_token = tokenizer.eos_token

# Tokenize prompts and targets
train_encodings = tokenizer([p[0] for p in prompts], truncation=True, padding=True, max_length=128, return_tensors="pt")
train_targets = tokenizer([p[1] for p in prompts], truncation=True, padding=True, max_length=10, return_tensors="pt")

# Ensure labels and inputs are properly aligned and non-empty
if train_encodings['input_ids'].shape[0] == 0 or train_targets['input_ids'].shape[0] == 0:
    raise ValueError("The input_ids or labels are empty. Please check the input data and tokenization.")

# Offset the labels to match causal language modeling requirements
labels = train_encodings['input_ids'].clone()
labels[labels == tokenizer.pad_token_id] = -100  # Mask padding tokens in labels for loss calculation

# Create DataLoader
train_data = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], labels)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
model.train()

# Fine-tuning loop
print("Starting fine-tuning with prompt-based inputs...")
for epoch in range(100):  # Set number of epochs as needed
    total_loss = 0
    for batch in train_loader:
        inputs, masks, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Forward pass with labels as target sequence
        outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Save the fine-tuned model (optional)
model.save_pretrained("fine_tuned_bloom_560m_prompt")
tokenizer.save_pretrained("fine_tuned_bloom_560m_prompt")
print("Fine-tuning complete.")

# Set model to evaluation mode
model.eval()


# Function to classify a sentence based on the prompt
def classify_sentence_with_prompt(sentence):
    prompt = create_prompt_input(sentence, "")[0]  # Generate prompt without target label
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 3, pad_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract "Yes" or "No" based on the model output
    if "Yes" in generated_text:
        return "Yes"
    elif "No" in generated_text:
        return "No"
    else:
        return generated_text


# Initialize counters for evaluation
e = 0
n = 0
y = 0
nc = 0
yc = 0

# Evaluation loop
with open("Results/resultsBasicTrainGPT.txt", "w") as file:
    file.write("Text" + "\t" + "class_label" + "\t" + "classification" + "\n")
    for i, r in test_dataset.iterrows():
        # Get classification result
        classification = classify_sentence_with_prompt(r.Text)

        # Update counters based on classification results
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

        # Write results to file
        file.write(r.Text + "\t" + r.class_label + "\t" + classification + "\n")

# Print evaluation summary
print("Number of errors:", e)
print("Number correct:", len(test_dataset) - e)
print("Number yes:", y)
print("Number no:", n)
print("Number correct yes:", yc)
print("Number correct no:", nc)
