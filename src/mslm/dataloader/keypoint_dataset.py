import h5py
import torch

class KeypointDataset():
    def __init__(self, h5Path, n_keypoints = 230, transform = None, return_label=False, max_length=5000):
        self.h5Path = h5Path
        self.n_keypoints = n_keypoints
        self.transform = transform

        self.return_label = return_label

        self.max_length = max_length

        self.processData()

    def processData(self):
        with h5py.File(self.h5Path, 'r') as f:
            datasets  = list(f.keys())

            self.valid_index = []

            for dataset in datasets:
                if dataset != "dataset2":
                    continue

                group  = list(f[dataset].keys())
                clip_ids  = list(f[dataset]["embeddings"].keys())

                for clip in clip_ids:
                    shape = f[dataset]["keypoints"][clip].shape[0]
                    if shape < self.max_length:
                        self.valid_index.append((dataset, clip))

    def __len__(self):
        return len(self.valid_index)
    
    def filter_unstable_keypoints_to_num(keypoints, keep_n=230):
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

        # clean noise 

        keypoint_normalized, _ = self.filter_unstable_keypoints_to_num(keypoint_normalized, keep_n=230)

        # print(keypoint.size())

        if self.return_label:
            return keypoint_normalized, torch.tensor(embedding), label

        return keypoint_normalized, torch.tensor(embedding), None
    
