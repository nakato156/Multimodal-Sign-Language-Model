import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, LOG=False):
    """
    batch: List of tuples (keypoints, embeddings)
      - keypoints: [T_i, K, D]
      - embeddings: [N_i, E]
    returns:
      keypoints_padded  → [B, T_max, K, D]
      frames_mask       → [B, T_max]     (True = padding)
      embeddings_padded → [B, N_max, E]
      embeddings_mask   → [B, N_max]     (True = padding)
    """
    keypoints_list  = [item[0] for item in batch]
    embeddings_list = [item[1] for item in batch]

    frame_lengths = torch.tensor([kp.size(0) for kp in keypoints_list],  dtype=torch.long)
    token_lengths = torch.tensor([emb.size(0) for emb in embeddings_list], dtype=torch.long)

    # pad_sequence pads on dim=0 up to the max length
    keypoints_padded  = pad_sequence(keypoints_list,  batch_first=True, padding_value=0.0)
    embeddings_padded = pad_sequence(embeddings_list, batch_first=True, padding_value=0.0)

    B, T_max, K, D = keypoints_padded.shape
    _, N_max, E    = embeddings_padded.shape

    arange_frames = torch.arange(T_max).unsqueeze(0).expand(B, -1)  # [B, T_max]
    arange_tokens = torch.arange(N_max).unsqueeze(0).expand(B, -1)  # [B, N_max]

    frames_mask     = arange_frames >= frame_lengths.unsqueeze(1)  # True = frame padding
    embeddings_mask = arange_tokens >= token_lengths.unsqueeze(1)  # True = token padding

    return keypoints_padded, frames_mask, embeddings_padded, embeddings_mask