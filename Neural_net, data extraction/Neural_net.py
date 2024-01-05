import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import numpy as np
import csv
import math

def import_data():
    folder_path = r'C:\Users\Lucas\Desktop\data'

    # Get a list of all files in the directory
    file_list = os.listdir(folder_path)

    # Filter out only the .npy files
    npy_files = [file for file in file_list if file.endswith('.npy')]

    # Loop through each .npy file and load its data
    skipped=0
    amount=0
    input=[]
    ground_truth=[]
    for npy_file in npy_files:
        file_path = os.path.join(folder_path, npy_file)
        loaded_data = (np.load(file_path)+40)/40
        print(f"Loaded data from {npy_file}")
        first_dash=npy_file.index('-')+1
        s_index=npy_file[first_dash:].index('-')+first_dash+1
        e_index=npy_file[s_index:].index('-')+s_index
        beats_file=npy_file[:s_index-1]+'.csv'
        start_time=int(npy_file[s_index:e_index])
        end_time=start_time+10000
        with open(r"C:\Users\Lucas\OneDrive\Eksamensprojekt\DataBeat"+'\\'+beats_file, 'r') as f:
            reader = csv.reader(f)
            row = next(reader)
        for i,data in enumerate(row):
            row[i] = int(data.strip('[] '))

        # Algorithm for finding the 5 beats before the endtime of the soundbite (could probably have been done easier), also sorts off bad data
        most=len(row)-1
        index=math.floor(len(row)/2)
        least=0
        amount+=1
        if row[4]>end_time or row[-1]<=end_time:
            skipped+=1
            continue
        else:
            while True:
                before = row[index]
                after = row[index+1]
                if before > end_time:
                    most=index
                    last=index
                    index=math.floor((index+least)/2)
                    if last==index:
                        index+=1
                elif after <= end_time:
                    least=index
                    last=index
                    index=math.floor((index+most)/2)
                    if last==index:
                        index+=1
                else:
                    break
        if row[index-4]<start_time:
            skipped+=1
            continue
        # End of algortihm
        # print(row[index-4],row[index-3],row[index-2],row[index-1],row[index])
        # print(row[index+1],"\n")
        intervals=[row[index-3]-row[index-4],row[index-2]-row[index-3],row[index-1]-row[index-2],row[index]-row[index-1],end_time-row[index]]
        input.append(loaded_data)
        ground_truth.append(intervals)

    print(f'{skipped}/{amount} = {round(skipped/amount*100,1)}%')
    return torch.tensor(np.array(input),requires_grad=False).view(-1,1,513,1723), torch.tensor(np.array(ground_truth),dtype=torch.float32,requires_grad=False)


keep_prob=1
class StaticBeatD(nn.Module):
    def __init__(self):
        super(StaticBeatD, self).__init__()
        # L1 ImgIn shape=(?, 513, 1723, 1)
        # Conv -> (?, 510, 1720, 32)
        # Pool -> (?, 255, 860, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 255, 860, 32)
        # Conv      ->(?, 252, 857, 64)
        # Pool      ->(?, 127, 429, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2,padding=1),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 127, 429, 64)
        # Conv ->(?, 124, 426, 128)
        # Pool ->(?, 62, 213, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 62x213x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(62 * 213 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 4 outputs
        self.fc2 = torch.nn.Linear(625, 4, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters

    def forward(self, x):
        out = self.layer1(x)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        print(out.shape)
        out = self.fc1(out)
        print(out.shape)
        out = self.fc2(out)
        print(out.shape)
        return out

# Create dataset from data
input, ground = import_data()
ground = ground/1000
dataset = TensorDataset(input[:200],ground[:200,:4])
batch_size = 12
trainset = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define network
net = StaticBeatD()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-16, momentum=0.9)
print(net(input[50].view(1,1,513,1723)))
print(net(input[51].view(1,1,513,1723)))

# For at loade et tidligere lavet netværk!

# loaded_net = StaticBeatD()  # Recreate the model architecture
# loaded_net.load_state_dict(torch.load(r'C:\Users\Lucas\Desktop\DTU\Git\iBeat\Neural_net, data extraction\your_model_state_dict.pth'))
# print(loaded_net(input[50].view(1,1,513,1723)))
# print(loaded_net(input[51].view(1,1,513,1723)))

net.train()
for epoch in range(15):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        print(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        # for name, param in net.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad)
        # time.sleep(100000)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 4 == 3:    # print every 4 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 4:.3f}')
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), r'C:\Users\Lucas\Desktop\DTU\Git\iBeat\Neural_net, data extraction\trained_network.pth')



"""for epoch in range(15):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')"""