import torch
import numpy as np

def load_prebuilt_dataset(batch_size=16, num_files=1):
    tracks, KDEs, PVs = [], [], []
    for i in range(1, num_files+1):
        tracks.append(torch.Tensor(np.load(f'/share/lazy/will/data/tracks_to_KDE/June30_2020_80k_{i}_tracks.h5.npy')))
        KDEs.append(torch.Tensor(np.load(f'/share/lazy/will/data/tracks_to_KDE/June30_2020_80k_{i}_KDEs.h5.npy')))
        PVs.append(torch.Tensor(np.load(f'/share/lazy/will/data/tracks_to_KDE/June30_2020_80k_{i}_PVs.h5.npy')))
# all 3
#     return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(tracks), torch.cat(KDEs), torch.cat(PVs)), batch_size=batch_size)
# tracks and KDEs
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(tracks), torch.cat(KDEs)), batch_size=batch_size)