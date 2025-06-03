import torch
from llama_cpp import Llama
import os

class LLM:
    def __init__(self, main_directory):
        MODEL_NAME = os.path.join(main_directory, "local_models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
        self.llm = Llama(model_path=MODEL_NAME, embedding=True)

    def run(self, label):
        embeddings = self.llm.create_embedding(label)
        t = torch.tensor(embeddings["data"][0]["embedding"])
        print(f"Embedding shape: {t.shape}")
        print(t)
        return t
