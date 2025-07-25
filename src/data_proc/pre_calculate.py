import torch
from llama_cpp import Llama
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self, main_directory):
        # self.MODEL_NAME = os.path.join(main_directory, "local_models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
        self.model_id = "unsloth/Llama-3.2-3B-Instruct-GGUF"
        self.filename = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.filename)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.filename)
        self.embeddings = self.model.get_input_embeddings()

    def unload_model(self):
        del self.model
        # del self.tokenizer
        self._cleanup()

    def _cleanup(self):
        torch.cuda.empty_cache()
        while gc.collect() != 0:
            break

    def run(self, label):
        input_ids = self.tokenizer(label, return_tensors="pt").input_ids.to('cuda')
        self.embeddings = self.embeddings.to('cuda')
        t = self.embeddings(input_ids[0])
        # t = torch.tensor(embeddings["data"][0]["embedding"]).to('cuda')
        print(f"Embedding shape: {t.shape}")
        return t
