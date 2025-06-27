import h5py
import torch
from torch.utils.data import random_split
import numpy as np
import random

class KeypointDataset():
    def __init__(self, h5Path, n_keypoints = 230, transform = None, return_label=False, max_length=5000, data_augmentation=True):
        self.h5Path = h5Path
        self.n_keypoints = n_keypoints
        self.transform = transform

        self.return_label = return_label

        self.max_length = max_length
        self.video_lengths = []
        self.data_augmentation = data_augmentation

        self.data_augmentation_dict = {
            0: "Original",
            1: "Length_variance",
            2: "Gaussian_jitter",
        }

        self.processData()

    def processData(self):
        with h5py.File(self.h5Path, 'r') as f:
            datasets  = list(f.keys())

            self.valid_index = []
            self.original_videos = []

            for dataset in datasets:
                #group  = list(f[dataset].keys())
                clip_ids  = list(f[dataset]["embeddings"].keys())

                for clip in clip_ids:
                    shape = f[dataset]["keypoints"][clip].shape[0]
                    if shape < self.max_length:
                        if self.data_augmentation:
                            for i in self.data_augmentation_dict:
                                self.valid_index.append((dataset, clip, i))
                                self.video_lengths.append(shape)
                            self.original_videos.append("")
                        else: 
                            self.valid_index.append((dataset, clip))
                            self.video_lengths.append(shape)

    def split_dataset(self, train_ratio):
        if self.data_augmentation:
            train_dataset, validation_dataset = random_split(self.original_videos, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))
            train_length = [self.video_lengths[i+x] 
                            for x in self.data_augmentation_dict 
                            for i in train_dataset.indices]
        else:
            train_dataset, validation_dataset = random_split(self, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))
            train_length = [self.video_lengths[i] for i in train_dataset.indices]
    
        val_length = [self.video_lengths[i] for i in validation_dataset.indices] 

        return train_dataset, validation_dataset, train_length, val_length

    def get_video_lengths(self):
        return self.video_lengths 
    
    def filter_unstable_keypoints_to_num(self, keypoints, keep_n):
        """
        Conserva los 'keep_n' keypoints más estables (con menor varianza temporal).
        """
        T, N, _ = keypoints.shape

        # Calcular varianza temporal por keypoint
        var = keypoints.var(dim=0).mean(dim=1)  # (N,)

        # Obtener los índices de los keypoints más estables
        _, indices = torch.topk(-var, k=keep_n)  # usamos -var para orden ascendente
        stable_mask = torch.zeros(N, dtype=torch.bool)
        stable_mask[indices] = True

        # Aplicar la máscara
        filtered = keypoints.clone()
        for i in range(N):
            if not stable_mask[i]:
                filtered[:, i] = 0

        return filtered, stable_mask

    def keypoint_normalization(self, keypoint):
        flat = keypoint.view(-1, 2)
        global_mins, _ = flat.min(dim=0)
        global_maxs, _ = flat.max(dim=0)

        global_ranges = global_maxs - global_mins
        global_ranges[global_ranges == 0] = 1.0

        gm = global_mins.unsqueeze(0).unsqueeze(0)
        gr = global_ranges.unsqueeze(0).unsqueeze(0)

        return  (keypoint - gm) / gr

    def length_variance(self, keypoint, scale_range=(0.8, 1.5)):
        T, J, C = keypoint.shape
        scale = random.uniform(*scale_range)
        T_new = int(round(T * scale))
        
        orig_times = np.linspace(0, T-1, num=T)
        new_times = np.linspace(0, T-1, num=T_new)

        flat = keypoint.reshape(T, J*C)
        streched = np.stack([
            np.interp(new_times, orig_times, flat[:, d])
            for d in range(J*C)
        ], axis = 1)
        keypoints_streched = streched.reshape(T_new, J, C)
        return keypoints_streched
    
    def guassian_jitter(self, keypoint, sigma=5.0, clip=3.0):
        keypoint_jitter = np.random.normal(loc=0.0, scale=sigma, size=keypoint.shape)

        if clip is not None:
            np.clip(keypoint_jitter, -clip, clip, out=keypoint_jitter)    
        
        return keypoint + keypoint_jitter

    def __len__(self):
        if self.data_augmentation:
            return int(len(self.valid_index)/len(self.data_augmentation_dict))
        else:
            return len(self.valid_index)

    def __getitem__(self, idx):
        mapped_idx = self.valid_index[idx]
            
        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[mapped_idx[0]]["keypoints"][mapped_idx[1]][:]
            embedding = f[mapped_idx[0]]["embeddings"][mapped_idx[1]][:]
        
            if self.return_label:
                label = f[mapped_idx[0]]["labels"][mapped_idx[1]][:][0].decode()

        if self.data_augmentation:
            if self.data_augmentation_dict[mapped_idx[2]] == "Gaussian_jitter":
                keypoint = self.guassian_jitter(keypoint)
                
            elif self.data_augmentation_dict[mapped_idx[2]] == "Length_variance":
                keypoint = self.length_variance(keypoint)

        #Keypoints a Tensor
        keypoint = torch.tensor(keypoint, dtype=torch.float32)

        # Keypoint Normalization
        keypoint_normalized = self.keypoint_normalization(keypoint)
        
        # clean noise 
        keypoint_normalized, _ = self.filter_unstable_keypoints_to_num(keypoint_normalized, self.n_keypoints)

        if self.return_label:
            return keypoint_normalized, torch.tensor(embedding), label

        return keypoint_normalized, torch.tensor(embedding), None