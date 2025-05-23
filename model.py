import torch
import torch.nn as nn

class DQCNN(nn.Module):
    def __init__(self, n_actions, n_obsSize, dropout_rate=0.2):
        super(DQCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = n_obsSize
        self.dropout_rate = dropout_rate

        # Calculate output dimensions after each convolutional layer
        h_in, w_in = self.input_size[1], self.input_size[2]
        
        # Conv1: (h-8)/4+1, (w-8)/4+1
        h_conv1 = (h_in - 8) // 4 + 1
        w_conv1 = (w_in - 8) // 4 + 1
        
        # Conv2: (h_conv1-4)/2+1, (w_conv1-4)/2+1
        h_conv2 = (h_conv1 - 4) // 2 + 1
        w_conv2 = (w_conv1 - 4) // 2 + 1
        
        # Conv3: (h_conv2-3)/1+1, (w_conv2-3)/1+1
        h_conv3 = (h_conv2 - 3) + 1
        w_conv3 = (w_conv2 - 3) + 1
        
        # Final flattened size for FC input
        self.fc_input_size = 64 * h_conv3 * w_conv3

        # Convolutional layers with dropout
        self.conv1 = nn.Conv2d(self.input_size[0], 32, kernel_size=8, stride=4)
        self.dropout2d_1 = nn.Dropout2d(p=dropout_rate/2)  # Lower rate for conv layers
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.dropout2d_2 = nn.Dropout2d(p=dropout_rate/2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dropout2d_3 = nn.Dropout2d(p=dropout_rate/2)
        
        self.flatten = nn.Flatten()
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(128, n_actions)
        # No dropout before final output

    def forward(self, x):
        # Convolutional layers with activation and dropout
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.dropout2d_1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.dropout2d_2(x)
        
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.dropout2d_3(x)
        
        x = self.flatten(x)
        
        # Fully connected layers with activation and dropout
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # Final output layer (no dropout)
        x = self.fc3(x)

        return x
    
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    img = Image.open("bluestacks_screenshot.png")
    np_img = np.array(img)
    gray_img = np.dot(np_img[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img = torch.unsqueeze(torch.tensor(gray_img), axis=0)  # Add channel dimension

    model = DQCNN(n_actions=4, n_obsSize=gray_img.shape, dropout_rate=0.2)
    print(model)

