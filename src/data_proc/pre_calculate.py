import torch
from llama_cpp import Llama
import os
import gc

class LLM:
    def __init__(self, main_directory):
        self.MODEL_NAME = os.path.join(main_directory, "local_models/Gemma-3n/gemma-3n-E2B-it-Q4_K_M.gguf")

    def load_model(self):
        self.llm = Llama(
            model_path=self.MODEL_NAME,
            embedding=True,
            n_gpu_layers=-1,
            verbose=True
        )

    def unload_model(self):
        del self.llm
        self._cleanup()

    def _cleanup(self):
        torch.cuda.empty_cache()
        while gc.collect() != 0:
            break

    def run(self, label):
        embeddings = self.llm.create_embedding(label)
        t = torch.tensor(embeddings["data"][0]["embedding"]).to('cuda')
        print(f"Embedding shape: {t.shape}")
        return t
