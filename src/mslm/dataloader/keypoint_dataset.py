import h5py
import torch
from torch.utils.data import random_split, Dataset, Subset, ConcatDataset
import numpy as np
import random

def scaling(keypoint):
    # Escalar los keypoints sin cambiar su cantidad
    scale = random.uniform(0.9, 1.1)
    return keypoint * scale

def horizontal_flip(keypoint):
    # Reflejar horizontalmente la secuencia de keypoints
    return keypoint.flip(dims=[1])  # Reflejo horizontal

def rotation_2D(keypoint):
    # Aseguramos que la rotación no cambie la cantidad de keypoints
    angle = random.uniform(-15, 15)  # Rotación aleatoria en grados
    angle_rad = torch.tensor(angle * np.pi / 180.0, dtype=torch.float32)  # Convertir a radianes
    rotation_matrix = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)], 
                                    [torch.sin(angle_rad), torch.cos(angle_rad)]], dtype=torch.float32)
    keypoint_rotated = torch.matmul(keypoint.view(-1, 2), rotation_matrix)
    return keypoint_rotated.view(keypoint.shape)

def length_variance(keypoint, scale=0.8):
    T, J, C = keypoint.shape
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

def gaussian_jitter(keypoint, sigma=0.0014, clip=3.0):
    keypoint_jitter = np.random.normal(loc=0.0, scale=sigma, size=keypoint.shape)

    if clip is not None:
        np.clip(keypoint_jitter, -clip, clip, out=keypoint_jitter)    
    
    return keypoint + keypoint_jitter

def apply_augmentation(keypoint, augmentation_type):
    """
    Aplica diferentes augmentaciones de acuerdo al tipo.
    """
    
    if augmentation_type == "Gaussian_jitter":
        return gaussian_jitter(keypoint)
    elif augmentation_type == "Length_variance":
        return length_variance(keypoint)
    elif augmentation_type == "Rotation_2D":
        return rotation_2D(keypoint)
    elif augmentation_type == "Horizontal_flip":
        return horizontal_flip(keypoint)
    elif augmentation_type == "Scaling":
        return scaling(keypoint)
    return keypoint

def keypoint_normalization(keypoint):
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

def filter_unstable_keypoints_to_num(keypoints, keep_n):
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

def normalize_augment_data(keypoint, augmentation_type, n_keypoints = 245):
    #Keypoints a Tensor
    if not isinstance(keypoint, torch.Tensor):
        keypoint = torch.as_tensor(keypoint, dtype=torch.float32)    

    # Clean noise 
    keypoint, _ = filter_unstable_keypoints_to_num(keypoint, n_keypoints)

    if augmentation_type != "Original":
        keypoint = apply_augmentation(keypoint, augmentation_type)
    
    #Keypoints a Tensor
    if not isinstance(keypoint, torch.Tensor):
        keypoint = torch.as_tensor(keypoint, dtype=torch.float32)    
    # Keypoint Normalization
    keypoint = keypoint_normalization(keypoint)

    keypoint.to(torch.float32)
    return keypoint

class TransformedSubset(Dataset):
    def __init__(self, subset: Subset, transform_fn: str, return_label=False, video_lengths=[], n_keypoints=245):
        self.subset    = subset
        self.transform = transform_fn
        self.return_label = return_label
        self.video_lengths = video_lengths
        self.n_keypoints = n_keypoints
        
        if self.transform == "Length_variance":
            self.video_lengths = [int(round(0.8 * video)) for video in self.video_lengths]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        keypoint, embedding, label = self.subset[idx]

        keypoint = normalize_augment_data(keypoint, self.transform, self.n_keypoints)

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding, dtype=torch.float32)
    
        embedding.to(torch.float32)

        if self.return_label:
            return keypoint, embedding, label

        return keypoint, embedding, None

class KeypointDataset(Dataset):
    def __init__(self, h5Path, n_keypoints=245, transform=None, return_label=False, max_length=5000, data_augmentation=True):
        self.h5Path = h5Path
        self.n_keypoints = n_keypoints
        self.transform = transform
        self.return_label = return_label
        self.max_length = max_length
        self.video_lengths = []
        self.data_augmentation = data_augmentation
    
        self.data_augmentation_dict = {
        #    0: "Length_variance",
            1: "Gaussian_jitter",
            2: "Rotation_2D",
            3: "Horizontal_flip",
            4: "Scaling"
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
                        self.valid_index.append((dataset, clip))
                        self.video_lengths.append(shape)
                
            self.dataset_length = len(self.valid_index)

    def split_dataset(self, train_ratio):
        train_dataset, validation_dataset = random_split(self, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))
        val_length = [self.video_lengths[i] for i in validation_dataset.indices] 
        
        if self.data_augmentation:

            train_length = [self.video_lengths[i] 
                            for i in train_dataset.indices]
            train_subset = Subset(self, train_dataset.indices)
            aug_subsets = [
                TransformedSubset(train_subset, 
                                  transform_fn=tf,
                                  return_label=False,
                                  video_lengths=train_length,
                                  n_keypoints=self.n_keypoints
                                  )
                for tf in self.data_augmentation_dict.values()
            ]

            trains_subset_length = [ length
                for subset in aug_subsets
                for length in subset.video_lengths
            ]
            
            train_lengths = train_length + trains_subset_length 
            train_dataset = ConcatDataset([train_subset, *aug_subsets])
            
            self.dataset_length = len(val_length) + len(train_length)
        else:
            train_lengths = [self.video_lengths[i] for i in train_dataset.indices]        

        print("Videos: ", self.dataset_length)
        return train_dataset, validation_dataset, train_lengths, val_length

    def get_video_lengths(self):
        return self.video_lengths 
    
    def __len__(self):
        return len(self.valid_index)

    def __getitem__(self, idx):
        mapped_idx = self.valid_index[idx]
            
        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[mapped_idx[0]]["keypoints"][mapped_idx[1]][:]
            embedding = f[mapped_idx[0]]["embeddings"][mapped_idx[1]][:]
        
            if self.return_label:
                label = f[mapped_idx[0]]["labels"][mapped_idx[1]][:][0].decode()

        keypoint = normalize_augment_data(keypoint, "Original", self.n_keypoints)

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding, dtype=torch.float32)
    
        embedding.to(torch.float32)

        if self.return_label:
            return keypoint, embedding, label

        return keypoint, embedding, None
