import torch
import torch.nn as nn
import torch.nn.functional as F

class DQCNN(nn.Module):
    def __init__(self, n_actions, n_obsSize, dropout_rate=0.1):
        super(DQCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Conv layers remain the same
        self.conv1 = nn.Conv2d(n_obsSize[0], 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Since we're using adaptive pooling to (4,4), the flattened size is fixed
        self.fc_input_size = 32 * 4 * 4  # = 512
        
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, n_actions)
        
        # Add bias toward NONE action (index 4)
        with torch.no_grad():
            # Initialize all biases to slightly negative values
            self.fc2.bias.fill_(-0.1)
            # Make the NONE action bias positive
            self.fc2.bias[4] = 0.5

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        
        # Adaptive pooling ensures consistent size regardless of input
        x = F.adaptive_avg_pool2d(x, (4, 4))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Debug info - uncomment if needed
        # print(f"Flattened shape: {x.shape}")
        
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    img = Image.open("bluestacks_screenshot.png")
    np_img = np.array(img)
    gray_img = np.dot(np_img[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img = torch.unsqueeze(torch.tensor(gray_img), axis=0)  # Add channel dimension

    model = DQCNN(n_actions=4, n_obsSize=gray_img.shape, dropout_rate=0.1)
    print(model)

