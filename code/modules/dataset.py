import random
import h5py as h5
from torch.utils.data import Dataset
import torch as t
import json


class MyDataset(Dataset):
    def __init__(self, h5_path, json_file, ids, load_dataset_to_gpu_ram):
        # Load data into RAM
        with h5.File(h5_path, "r") as f:
            self.inputs = {
                id_: t.tensor(f[id_ + "-512-256.npy"][()], dtype=t.float32)
                for id_ in ids
            }

        self.bins = {
            id_: t.tensor(count, dtype=t.float32)
            for id_, count in json.load(open(json_file, "r")).items()
        }
        self.ids = list(self.inputs.keys())

        # Transfer data to GPU memory if possible
        if load_dataset_to_gpu_ram and t.cuda.is_available():
            print("Sending dataset to cuda")
            self.inputs = {id_: spect.cuda() for id_, spect in self.inputs.items()}
            self.bins = {id_: bin.cuda() for id_, bin in self.bins.items()}
        else:
            print("Could not send dataset to cuda")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        return self.inputs[id_], self.bins[id_]


def find_divide_sets(path_json, split_size, seed=1000):
    random.seed(seed)
    with open(path_json, "r") as file:
        a = json.load(file).keys()
    songs = set()
    for key in a:
        index1 = key.index("-")
        index2 = key[index1 + 1 :].index("-") + index1 + 1
        songs.add(key[:index2])

    train_size = round(split_size * len(songs))
    test_size = len(songs) - train_size

    train_songs = random.sample(list(sorted(songs)), train_size)
    test_songs = songs - set(train_songs)

    train_ids = []
    test_ids = []
    for key in a:
        index1 = key.index("-")
        index2 = key[index1 + 1 :].index("-") + index1 + 1
        if key[:index2] in train_songs:
            train_ids.append(key)
        elif key[:index2] in test_songs:
            test_ids.append(key)
        else:
            print("something went wrong while dividing test and training sets")

    return train_ids, test_ids
