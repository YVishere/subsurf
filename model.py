import torch
import torch.nn as nn
import torch.nn.functional as F

class DQCNN(nn.Module):
    def __init__(self, n_actions, n_obsSize):
        super(DQCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convolutional layers (deeper, as suggested)
        self.conv1 = nn.Conv2d(n_obsSize[0], 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Adaptive pooling to (4,4)
        self.fc_input_size = 64 * 4 * 4

        # Dueling streams
        self.value_fc = nn.Linear(self.fc_input_size, 256)
        self.value_out = nn.Linear(256, 1)
        self.advantage_fc = nn.Linear(self.fc_input_size, 256)
        self.advantage_out = nn.Linear(256, n_actions)

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01, inplace=True)
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(x.size(0), -1)

        value = F.leaky_relu(self.value_fc(x), negative_slope=0.01, inplace=True)
        value = self.value_out(value)
        advantage = F.leaky_relu(self.advantage_fc(x), negative_slope=0.01, inplace=True)
        advantage = self.advantage_out(advantage)
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals
    
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    img = Image.open("bluestacks_screenshot.png")
    np_img = np.array(img)
    gray_img = np.dot(np_img[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img = torch.unsqueeze(torch.tensor(gray_img), axis=0)  # Add channel dimension

    model = DQCNN(n_actions=4, n_obsSize=gray_img.shape)
    print(model)

