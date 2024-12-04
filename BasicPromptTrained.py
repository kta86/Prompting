import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")
torch.cuda.empty_cache()

# Load datasets
train_dataset = pd.read_csv("Data/CT24_checkworthy_english_dev.tsv", dtype=object, encoding="utf-8", sep='\t')
test_dataset = pd.read_csv("Data/CT24_checkworthy_english_dev-test1.tsv", dtype=object, encoding="utf-8", sep='\t')

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)

# Set pad token (using eos token if pad token not available)
tokenizer.pad_token = tokenizer.eos_token


# Function to create prompt and target text
def create_prompt_input(sentence, label):
    prompt = (f"A check-worthy sentence contains a claim that can be fact-checked. If a sentence is check-worthy, "
              f"classify it as Yes, if not classify it as No. Classify the following sentences into one of the " 
              f"following categories only: Yes, No. Classify the following sentence:\n"
              f"Sentence: \"{sentence}\"\nClassification: ")
    target_text = f" {label}"
    return prompt, target_text


# Prepare training data prompts and labels
train_texts = train_dataset['Text'].tolist()
train_labels = train_dataset['class_label'].apply(lambda x: "Yes" if x == "Yes" else "No").tolist()
prompts = [create_prompt_input(text, label) for text, label in zip(train_texts, train_labels)]

# Tokenize prompts and labels
train_encodings = tokenizer([p[0] for p in prompts], truncation=True, padding=True, max_length=128, return_tensors="pt")
train_targets = tokenizer([p[1] for p in prompts], truncation=True, padding=True, max_length=3, return_tensors="pt")

# Ensure labels and inputs are properly aligned
labels = train_encodings['input_ids'].clone()
labels[labels == tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

# DataLoader creation
train_data = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], labels)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()

# Training loop
print("Starting fine-tuning with prompt-based inputs...")
for epoch in range(3):  # Set the appropriate number of epochs
    total_loss = 0
    for batch in train_loader:
        inputs, masks, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()

        # Forward pass with labels
        outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Save the fine-tuned model and tokenizer
model_save_path = "fine_tuned_gpt2_prompt"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print("Fine-tuning complete and model saved.")

# Load the model for evaluation
model = AutoModelForCausalLM.from_pretrained(model_save_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model.eval()



# Function to classify a sentence using the fine-tuned model
def classify_sentence_with_prompt(sentence):
    prompt = create_prompt_input(sentence, "")[0]  # Generate prompt without target
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    forced_tokens = tokenizer(["Yes", "No"], return_tensors="pt")['input_ids']
    output = model.generate(
        **inputs,
        max_length=inputs['input_ids'].shape[1] + 10,  # Expect only "Yes" or "No"
        pad_token_id=tokenizer.eos_token_id,
        forced_bos_token_id=forced_tokens[0][0]  # Forces either "Yes" or "No"
    )

    #output = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 50, pad_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    classified = generated_text.split('Classification:')[-1].strip().split()[0]
    if "Yes" in classified:
        return "Yes"
    elif "No" in classified:
        return "No"
    else:
        return "unknown"


for i, r in test_dataset.iterrows():
    if i >= 5:  # Limit to 5 examples for inspection
        break
    sentence = r['Text']
    expected = r['class_label']
    classification = classify_sentence_with_prompt(sentence)

    print(f"Sentence: {sentence}")
    print(f"Expected: {expected}")
    print(f"Generated: {classification}")
    print()

# Initialize counters for evaluation metrics
e, n, y, nc, yc = 0, 0, 0, 0, 0

# Evaluation loop
with open("Results/resultsBasicTrainGPT.txt", "w", encoding="utf-8") as file:
    file.write("Text\tclass_label\tclassification\n")
    for _, r in test_dataset.iterrows():
        classification = classify_sentence_with_prompt(r.Text)

        # Update counters based on result
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

        file.write(f"{r.Text}\t{r.class_label}\t{classification}\n")

# Print evaluation summary
print("Number of errors:", e)
print("Number correct:", len(test_dataset) - e)
print("Number yes:", y)
print("Number no:", n)
print("Number correct yes:", yc)
print("Number correct no:", nc)
