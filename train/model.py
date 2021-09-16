import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size):
        super(CNN, self).__init__()
        conv0 = nn.Conv2d(1, num_maps1, (8, 16), padding=(4, 0), stride=(2, 2), bias=True)
        pool = nn.MaxPool2d(2)
        conv1 = nn.Conv2d(num_maps1, num_maps2, (5, 5), padding=2, stride=(2, 1), bias=True)
        self.num_hidden_input = num_hidden_input
        self.encoder1 = nn.Sequential(conv0, nn.ReLU(), pool, nn.BatchNorm2d(num_maps1, affine=True))
        self.encoder2 = nn.Sequential(conv1, nn.ReLU(), pool, nn.BatchNorm2d(num_maps2, affine=True))
        self.output = nn.Sequential(
            nn.Linear(num_hidden_input, hidden_size), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size, num_labels)
        )

    def forward(self, x):
        x = x[:, :1]  # log_mels only
        x = x.permute(0, 1, 3, 2)  # (time, n_mels)
        # pass through first conv layer
        x1 = self.encoder1(x)
        # pass through second conv layer
        x2 = self.encoder2(x1)
        # flattening - keep first dim batch same, flatten last 3 dims
        x = x2.view(-1, self.num_hidden_input)
        return self.output(x)
