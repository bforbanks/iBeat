import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import time
import numpy as np
import csv
import math
import json
from tqdm import tqdm

keep_prob = 1
base_path = r"C:\Users\Lucas\OneDrive - Private\OneDrive\Eksamensprojekt"

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, annotations_file_path, data_dir, transform=None, target_transform=None):
        self.beat_labels = pd.read_csv(annotations_file_path, sep=";")
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__ (self):
        return len(self.beat_labels)

    def __getitem__(self, idx):
      sound_path = os.path.join(self.data_dir, self.beat_labels.iloc[idx, 0] + '-512-256.npy')

      ## read the file
      sound = torch.from_numpy((np.load(sound_path, allow_pickle=True) + 40) / 40)
      label = torch.tensor(self.beat_labels.iloc[idx, 1:] / 1000)

      if self.transform:
          sound = self.transform(sound)

      if self.target_transform:
          label = self.target_transform(label)

      return sound, label
    


class StaticBeatD(nn.Module):
    def __init__(self):
        super().__init__()
        super(StaticBeatD, self).__init__()
        # [257, 1723]
        # L1 ImgIn shape=(?, 257, 1723, 1)
        # Conv -> (?, 254, 1720, 32)
        # Pool -> (?, 127, 860, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob),
        )
        # L2 ImgIn shape=(?, 127, 860, 32)
        # Conv      ->(?, 124, 857, 64)
        # Pool      ->(?, 63, 429, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob),
        )
        # L3 ImgIn shape=(?, 63, 429, 64)
        # Conv ->(?, 60, 426, 128)
        # Pool ->(?, 30, 213, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Dropout(p=1 - keep_prob),
        )

        # L4 FC 30x213x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(30*213*128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1, torch.nn.ReLU(), torch.nn.Dropout(p=1 - keep_prob)
        )

        # L5 Final FC 625 inputs -> 4 outputs
        self.fc2 = torch.nn.Linear(625, 4, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)  # initialize parameters

    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.fc2(out)
        #print(out.shape)
        return out


# Create dataset from data
train_dataset = CustomDataset(base_path + "/snippet_intervals.csv", base_path + '/DataSpectrogram-512')
train_dataloader = DataLoader(train_dataset, batch_size = 24, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(train_features[0])
print(train_labels[0])

# Define network
net = StaticBeatD()
# net.load_state_dict(torch.load(base_path + '/test.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adadelta(net.parameters())

net.to(device)
net.train()

## Epoch loop
for i2, epoch in enumerate(tqdm(range(200), desc= "Epochs")):  # loop over the dataset multiple times
    running_loss = 0.0
    ## Batch loop
    for i, (inputs, labels) in tqdm(enumerate(train_dataloader), desc = "Batches"):
        labels=labels[:,:4]; labels=labels.to(torch.float32)
        inputs=inputs.view(-1,1,257,1723); inputs=inputs.to(torch.float32)
        inputs, labels = inputs.to(device), labels.to(device)
        # print(f"Feature batch shape: {inputs[j].size()}")
        # print(f"Labels batch shape: {labels[j].size()}")
        # print(inputs[j])
        # print(labels[j])

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 24 == 23:  # print every 4 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 24:.3f}")
            running_loss = 0.0
    torch.save(net.state_dict(), r"C:\Users\Lucas\Desktop\DTU\Git\iBeat\Neural_net, data extraction\Nets\Adadelta" + f"/testAdadelta{i2}.pth",)
print("Finished Training")



# loaded_net = StaticBeatD()  # Recreate the model architecture
# loaded_net.load_state_dict(torch.load(base_path + '/test.pth'))
# print(loaded_net(input[1].view(1,1,257,1723))*1000)
# print(ground[1])
# print(loaded_net(input[2].view(1,1,257,1723))*1000)
# print(ground[2])
