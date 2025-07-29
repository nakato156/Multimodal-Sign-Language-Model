import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self, model_id="unsloth/Llama-3.2-3B-Instruct"):
        self.model_id = model_id

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
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
        return t
