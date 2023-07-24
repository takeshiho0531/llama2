from transformers import AutoModel, AutoTokenizer
import torch

model_name="meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Hello, how are you?"
tokens = tokenizer(text, return_tensors="pt")

print("text: ", text)
print("tokens: ", tokens)

with torch.no_grad():
    output = model(**tokens)
    embeddings = output.last_hidden_state

print("embeddings.shape: ", embeddings.shape)
print("embeddings: ", embeddings)