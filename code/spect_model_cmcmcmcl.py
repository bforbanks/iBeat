import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Architecture(nn.Module):
    def __init__(self, actual_dropout_rate=0.9):
        super(Architecture, self).__init__()

        self.code = "cmcmcmc_1L"

        # Settings
        # hparams
        self.BATCH_SIZE = [32]  # cycleable
        self.DATA_SHUFFLE = True
        self.LEARNING_RATE = [0.001]  # cycleable
        self.SCHEDULE_FACTOR = 0.1
        self.SCHEDULE_PATIENCE = 1
        self.DESIRED_LOSS = 0.04
        self.TEST_INCREASE_LIMIT = 7
        self.MIN_EPOCHS = 10
        self.SPLIT_SIZE = 0.9
        self.DROPOUT_RATE = [0.9]  # cycleable
        # "DataSpectrogram-512-4x-mean" "Wav"
        self.DATA_SET = "DataSpectrogram"
        self.DATA_HEIGHT = 64
        self.DATA_WIDTH = 430
        self.TARGET_SET = "jfdklsajfe"
        self.TARGET_LEN = 430

        self.actual_dropout_rate = actual_dropout_rate
        # Architecture
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(
            in_features=128 * 8 * 53,
            out_features=430,
        )
        self.dropout = nn.Dropout(self.actual_dropout_rate)

    def forward(self, x):
        x = x.view(-1, 1, 64, 430)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 128 * 8 * 53)
        x = self.dropout(x)
        x = t.sigmoid(self.fc1(x))
        return x

    def init_criterion_optimizer(self, learning_rate):
        criterion = nn.BCELoss()  # cycleable
        # cycleable
        optimizer = [optim.Adam(self.parameters(), lr=learning_rate)]
        return (criterion, optimizer)

    def init_scheduler(self, optimizer):
        return ReduceLROnPlateau(
            optimizer,
            "min",
            patience=self.SCHEDULE_PATIENCE,
            factor=self.SCHEDULE_FACTOR,
            verbose=True,
        )
