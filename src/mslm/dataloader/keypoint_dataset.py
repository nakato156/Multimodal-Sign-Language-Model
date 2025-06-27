import h5py
import torch
from torch.utils.data import random_split
import random
import numpy as np

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

        self.processData()

    def processData(self):
        with h5py.File(self.h5Path, 'r') as f:
            datasets  = list(f.keys())

            self.valid_index = []

            for dataset in datasets:
                clip_ids  = list(f[dataset]["embeddings"].keys())

                for clip in clip_ids:
                    shape = f[dataset]["keypoints"][clip].shape[0]
                    if shape < self.max_length:
                        if self.data_augmentation:
                            for i in self.data_augmentation_dict:
                                self.valid_index.append((dataset, clip, i))
                                self.video_lengths.append(shape)
                        else: 
                            self.valid_index.append((dataset, clip))
                            self.video_lengths.append(shape)

    def split_dataset(self, train_ratio):
        train_dataset, validation_dataset = random_split(self, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))

        if self.data_augmentation:
            train_length = [self.video_lengths[i+x] 
                            for x in self.data_augmentation_dict 
                            for i in train_dataset.indices]
        else:
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

    def gaussian_jitter(self, keypoint):
        noise = torch.randn_like(keypoint) * 0.01  # Ruido gaussiano
        return keypoint + noise

    def length_variance(self, keypoint):
        """
        Aumenta o disminuye la amplitud del movimiento de las articulaciones de forma aleatoria,
        sin cambiar la cantidad de keypoints ni su secuencia temporal.
        """
        # Generar un factor de escala aleatorio para los movimientos de las articulaciones
        scale = random.uniform(0.8, 1.2)
        
        # Modificar las coordenadas de los keypoints multiplicando por el factor de escala
        keypoint_scaled = keypoint * scale
        
        return keypoint_scaled


    def rotation_2D(self, keypoint):
        # Aseguramos que la rotación no cambie la cantidad de keypoints
        angle = random.uniform(-15, 15)  # Rotación aleatoria en grados
        angle_rad = np.deg2rad(angle)
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
        return len(self.valid_index)

    def __getitem__(self, idx):
        if self.data_augmentation:
            mapped_idx = self.valid_index[idx]
        else:
            mapped_idx = self.valid_index[idx]
            
        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[mapped_idx[0]]["keypoints"][mapped_idx[1]][:]
            embedding = f[mapped_idx[0]]["embeddings"][mapped_idx[1]][:]
        
            if self.return_label:
                label = f[mapped_idx[0]]["labels"][mapped_idx[1]][:][0].decode()

        # Keypoints a Tensor
        keypoint = torch.tensor(keypoint, dtype=torch.float32)

        if self.data_augmentation:
            augmentation_type = self.data_augmentation_dict[mapped_idx[2]]
            keypoint = self.apply_augmentation(keypoint, augmentation_type)

        # Keypoint Normalization
        keypoint_normalized = self.keypoint_normalization(keypoint)
        
        # Clean noise 
        keypoint_normalized, _ = self.filter_unstable_keypoints_to_num(keypoint_normalized, self.n_keypoints)

        if self.return_label:
            return keypoint_normalized, torch.tensor(embedding), label

        return keypoint_normalized, torch.tensor(embedding), None
