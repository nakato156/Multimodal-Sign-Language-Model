import torch
from torch.utils.data import Dataset

class SignDataLoader(Dataset):
    def __init__(self, llama_tokenizer, keypointReader):
        self.llama_tokenizer = llama_tokenizer
        self.keypointReader = keypointReader
        # self.llama_embed_layer = llama_embed_layer
        self.max_len = 128

    def __getitem__(self, idx):
        data, label = self.keypointReader[idx]
        input_ids = self.llama_tokenizer(label)["input_ids"]

        input_ids = torch.tensor(input_ids,requires_grad=False)
        input_ids_pad = torch.full((self.max_len - input_ids.shape[0],),128004)
        input_ids = torch.cat([input_ids, input_ids_pad])

        # embeddings = self.llama_embed_layer(input_ids)

        #print(data.shape)
        #print(embeddings.shape)        

        return data, input_ids

    def __len__(self):
        return len(self.keypointReader)