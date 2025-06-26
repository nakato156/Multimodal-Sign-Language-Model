import torch
from llama_cpp import Llama
import os
import gc

class LLM:
    def __init__(self, main_directory):
        self.MODEL_NAME = os.path.join(main_directory, "local_models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf")

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


if __name__ == "__main__":
    main_directory = os.path.dirname(os.path.abspath(__file__))
    main_directory = "/home/giorgio6846/Code/Sign-AI"
    llm = LLM(main_directory)
    llm.load_model()
    
    label = "Example text for embedding"
    embedding = llm.run(label)
    
    print(f"Generated embedding: {embedding}")
    
    llm.unload_model()
    print("Model unloaded and resources cleaned up.")