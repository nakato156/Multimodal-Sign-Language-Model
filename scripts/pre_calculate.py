import torch
from llama_cpp import Llama

MODEL_NAME = "../local_models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

def run(llm: Llama):
    embeddings = llm.create_embedding("Hello, world!")
    t = torch.tensor(embeddings["data"][0]["embedding"])
    print(f"Embedding shape: {t.shape}")

if __name__ == "__main__":
    llm = Llama(model_path=MODEL_NAME, embedding=True)
    run(llm)