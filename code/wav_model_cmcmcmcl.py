import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Architecture(nn.Module):
    def __init__(self, actual_dropout_rate=0.9):
        super(Architecture, self).__init__()

        self.code = "wav_cmcmcmcl"

        # Settings
        # hparams
        self.BATCH_SIZE = [32]  # cycleable
        self.DATA_SHUFFLE = True
        self.LEARNING_RATE = [1]  # cycleable
        self.SCHEDULE_FACTOR = 0.1
        self.SCHEDULE_PATIENCE = 1
        self.DESIRED_LOSS = 0.04
        self.TEST_INCREASE_LIMIT = 7
        self.MIN_EPOCHS = 10
        self.SPLIT_SIZE = 0.9
        self.DROPOUT_RATE = [0.9]  # cycleable
        # "DataSpectrogram-512-4x-mean" "Wav"
        self.DATA_SET = "Wav"
        self.DATA_HEIGHT = 64
        self.DATA_WIDTH = 430
        self.TARGET_SET = "jfdklsajfe"
        self.TARGET_LEN = 430

        self.actual_dropout_rate = actual_dropout_rate

        # Architecture
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.conv4 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=128 * (27520 // 64),
                             out_features=self.TARGET_LEN)
        self.dropout = nn.Dropout(self.actual_dropout_rate)

    def forward(self, x):
        x = x.view(-1, 1, 27520)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 128 * (27520 // 64))
        x = self.dropout(x)
        x = t.sigmoid(self.fc1(x))
        return x

    def init_criterion_optimizer(self, learning_rate):
        criterion = nn.BCELoss()  # cycleable
        optimizer = [optim.Adadelta(
            self.parameters(), lr=learning_rate)]  # cycleable
        return (criterion, optimizer)

    def init_scheduler(self, optimizer):
        return ReduceLROnPlateau(
            optimizer,
            "min",
            patience=self.SCHEDULE_PATIENCE,
            factor=self.SCHEDULE_FACTOR,
            verbose=True,
        )
