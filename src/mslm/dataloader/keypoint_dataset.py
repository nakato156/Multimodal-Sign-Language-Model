import h5py
import torch
from torch.utils.data import random_split, Dataset, Subset, ConcatDataset
import numpy as np
import random

class TransformedSubset(Dataset):
    def __init__(self, subset: Subset, transform_fn: callable, return_label=False):
        self.subset    = subset
        self.transform = transform_fn
        self.return_label = return_label

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        keypoints, embedding, label = self.subset[idx]

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding, dtype=torch.float32)

        if self.return_label:
            return keypoints, embedding, label

        return keypoints, embedding, None

class KeypointDataset():
    def __init__(self, h5Path, n_keypoints=230, transform=None, return_label=False, max_length=5000, data_augmentation=True):
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
            3: "Rotation_2D",
            4: "Horizontal_flip",
            5: "Scaling"
        }

        self.dataset_length = 0
        self.processData()

    def processData(self):
        with h5py.File(self.h5Path, 'r') as f:
            datasets  = list(f.keys())
            datasets = sorted(datasets)
        
            self.valid_index = []
            self.original_videos = []

            for dataset in datasets:
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
                
            self.dataset_length = len(self.valid_index)

    def split_dataset(self, train_ratio):
        if self.data_augmentation:
            train_dataset, validation_dataset = random_split(self.original_videos, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))
            train_length = [self.video_lengths[i+x] 
                            for x in self.data_augmentation_dict 
                            for i in train_dataset.indices]

            train_subset = Subset(self, train_dataset.indices)
            aug_subsets = [
                TransformedSubset(train_subset, tf)
                for tf in self.data_augmentation_dict.values()
            ]
            
            train_dataset = ConcatDataset([train_subset, *aug_subsets])
            
        else:
            train_dataset, validation_dataset = random_split(self.original_videos, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))
            train_length = [self.video_lengths[i] for i in train_dataset.indices]
    
        val_length = [self.video_lengths[i] for i in validation_dataset.indices] 
        self.dataset_length = len(val_length) + len(train_length)

        print("Videos: ", self.dataset_length)
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
        keypoint = abs(keypoint)

        mask_keypoints = ~((keypoint[...,0] < 5) & (keypoint[...,1] < 5))
        
        valid_points = keypoint[mask_keypoints].view(-1, 2)
        if valid_points.numel() > 0:
            global_mins, _ = valid_points.min(dim=0)
            global_maxs, _ = valid_points.max(dim=0)
        else:
            global_mins, _ = torch.zeros(2, device=keypoint.device)
            global_maxs, _ = torch.ones(2, device=keypoint.device)
                        
        global_ranges = global_maxs - global_mins
        global_ranges[global_ranges == 0] = 1.0

        normalized = (keypoint - global_mins) / global_ranges

        normalized[~mask_keypoints] = keypoint[~mask_keypoints] 

        return normalized
    
    
    def apply_augmentation(self, keypoint, augmentation_type):
        """
        Aplica diferentes augmentaciones de acuerdo al tipo.
        """
        if augmentation_type == "Gaussian_jitter":
            return self.gaussian_jitter(keypoint)
        elif augmentation_type == "Length_variance":
            return self.length_variance(keypoint)
        elif augmentation_type == "Rotation_2D":
            return self.rotation_2D(keypoint)
        elif augmentation_type == "Horizontal_flip":
            return self.horizontal_flip(keypoint)
        elif augmentation_type == "Scaling":
            return self.scaling(keypoint)
        return keypoint

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
    
    def gaussian_jitter(self, keypoint, sigma=5.0, clip=3.0):
        keypoint_jitter = np.random.normal(loc=0.0, scale=sigma, size=keypoint.shape)

        if clip is not None:
            np.clip(keypoint_jitter, -clip, clip, out=keypoint_jitter)    
        
        return keypoint + keypoint_jitter
    
    def rotation_2D(self, keypoint):
        # Aseguramos que la rotación no cambie la cantidad de keypoints
        angle = random.uniform(-15, 15)  # Rotación aleatoria en grados
        angle_rad = torch.tensor(angle * np.pi / 180.0, dtype=torch.float32)  # Convertir a radianes
        rotation_matrix = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)], 
                                       [torch.sin(angle_rad), torch.cos(angle_rad)]], dtype=torch.float32)
        keypoint_rotated = torch.matmul(keypoint.view(-1, 2), rotation_matrix)
        return keypoint_rotated.view(keypoint.shape)

    def horizontal_flip(self, keypoint):
        # Reflejar horizontalmente la secuencia de keypoints
        return keypoint.flip(dims=[1])  # Reflejo horizontal

    def scaling(self, keypoint):
        # Escalar los keypoints sin cambiar su cantidad
        scale = random.uniform(0.9, 1.1)
        return keypoint * scale

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        mapped_idx = self.valid_index[idx]
            
        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[mapped_idx[0]]["keypoints"][mapped_idx[1]][:]
            embedding = f[mapped_idx[0]]["embeddings"][mapped_idx[1]][:]
        
            if self.return_label:
                label = f[mapped_idx[0]]["labels"][mapped_idx[1]][:][0].decode()

        if self.data_augmentation:
            keypoint = torch.tensor(keypoint, dtype=torch.float32)
            
            augmentation_type = self.data_augmentation_dict[mapped_idx[2]]
            keypoint = self.apply_augmentation(keypoint, augmentation_type)
        
        #Keypoints a Tensor
        if not isinstance(keypoint, torch.Tensor):
            keypoint = torch.as_tensor(keypoint, dtype=torch.float32)    
    
        # Clean noise 
        #keypoint, _ = self.filter_unstable_keypoints_to_num(keypoint, self.n_keypoints)

        # Keypoint Normalization
        keypoint_normalized = self.keypoint_normalization(keypoint)

        if self.return_label:
            return keypoint_normalized, torch.tensor(embedding), label

        return keypoint_normalized, torch.tensor(embedding), None
