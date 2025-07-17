from torch.utils.data import Sampler
from transformers.trainer_pt_utils import get_length_grouped_indices

class BatchSampler(Sampler):
    def __init__(self, video_dataset, batch_size: int):
        self.batches = self._grouped_indices(video_dataset, batch_size)

    def _grouped_indices(self, video_dataset, batch_size):
        sampler = get_length_grouped_indices(
            lengths=video_dataset,
            batch_size=batch_size
        )

        batches = [sampler[i:i+batch_size] for i in range(0, len(sampler), batch_size)]
        return batches

    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)