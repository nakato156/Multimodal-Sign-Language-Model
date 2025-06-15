import h5py
import torch

class KeypointDataset():
    def __init__(self, h5Path, transform = None, return_label=False):
        self.h5Path = h5Path
        self.transform = transform

        self.return_label = return_label

        self.processData()

    def processData(self):
        with h5py.File(self.h5Path, 'r') as f:
            datasets  = list(f.keys())

            self.valid_index = []

            for dataset in datasets:
                group  = list(f[dataset].keys())
                clip_ids  = list(f[dataset]["embeddings"].keys())

                for clip in clip_ids:
                    self.valid_index.append((dataset, clip))

    def __len__(self):
        return len(self.valid_index)

    def __getitem__(self, idx):
        mapped_idx = self.valid_index[idx]

        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[mapped_idx[0]]["keypoints"][mapped_idx[1]][:]
            embedding = f[mapped_idx[0]]["embeddings"][mapped_idx[1]][:]
        
            if self.return_label:
                label = f[mapped_idx[0]]["labels"][mapped_idx[1]][:][0].decode()

        #Keypoints a Tensor
        keypoint = torch.tensor(keypoint, dtype=torch.float32)

        flat = keypoint.view(-1, 2)
        global_mins, _ = flat.min(dim=0)
        global_maxs, _ = flat.max(dim=0)

        global_ranges = global_maxs - global_mins
        global_ranges[global_ranges == 0] = 1.0

        gm = global_mins.unsqueeze(0).unsqueeze(0)
        gr = global_ranges.unsqueeze(0).unsqueeze(0)

        keypoint_normalized = (keypoint - gm) / gr

        # print(keypoint.size())

        if self.return_label:
            return keypoint_normalized, torch.tensor(embedding), label

        return keypoint_normalized, torch.tensor(embedding), None