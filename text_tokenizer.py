from transformers import AutoTokenizer, AutoModel
import torch
import os, json

# 1️⃣ Load ParsBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

# 2️⃣ Set folder paths
input_folder = "../text"       # folder containing txt files
output_folder = "../tokenized_embeddings"  # output folder
os.makedirs(output_folder, exist_ok=True)

# 3️⃣ Initialize vocabulary counter
vocabulary = set()
total_tokens_count = 0
files_processed = 0

# 4️⃣ Iterate over files
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Update vocabulary and counters
        vocabulary.update(tokens)  # Add unique tokens to vocabulary set
        total_tokens_count += len(tokens)  # Count total tokens across all files
        files_processed += 1

        # Print tokens and IDs
        print(f"\n📄 File: {filename}")
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)

        # Convert to tensor and pass through the model
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)

        # Token embeddings
        token_embeddings = outputs.last_hidden_state[0]  # [seq_len, 768]

        # Sentence embedding (mean of token embeddings)
        sentence_embedding = torch.mean(token_embeddings, dim=0)

        # Print embedding shapes
        print("Token embeddings shape:", token_embeddings.shape)
        print("Sentence embedding shape:", sentence_embedding.shape)

        # Save everything to a JSON file
        output_path = os.path.join(output_folder, filename.replace(".txt", ".json"))
        json.dump({
            "text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "token_embeddings": token_embeddings.detach().tolist(),
            "sentence_embedding": sentence_embedding.detach().tolist()
        }, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 5️⃣ Print vocabulary statistics
print("\n" + "="*50)
print("📊 VOCABULARY STATISTICS")
print("="*50)
print(f"Total files processed: {files_processed}")
print(f"Total tokens across all files: {total_tokens_count}")
print(f"Unique tokens (vocabulary size): {len(vocabulary)}")
print(f"Average tokens per file: {total_tokens_count / files_processed if files_processed > 0 else 0:.2f}")

# Optional: Save vocabulary to a file
vocabulary_file = os.path.join(output_folder, "vocabulary.txt")
with open(vocabulary_file, "w", encoding="utf-8") as f:
    for token in sorted(vocabulary):
        f.write(token + "\n")
print(f"Vocabulary saved to: {vocabulary_file}")

print("\n✅ All files processed and saved in", output_folder)