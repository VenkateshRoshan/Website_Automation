
# Importing required libraries

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F

class CNNRNNModel(nn.Module):
    def __init__(self, num_classes=36,max_len=6):  # Change to 63
        super(CNNRNNModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layer to transform CNN output to RNN input
        self.fc1 = nn.Linear(256 * 5 * 12, 512*max_len)  # Adjust input size based on final CNN feature map size
        
        # RNN layers
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Final fully connected layers
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # CNN forward pass
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # print(x.shape)

        # Flatten the output from the CNN
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc1(x)

        # print(x.shape)
        
        # Reshape for RNN input (batch_size, seq_length, input_size)
        x = x.view(x.size(0), 6, 512)  # Ensure the sequence length is 6
        
        # RNN forward pass
        x, _ = self.rnn(x)
        
        # Apply fully connected layers to each time step
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        
        return x

class Image2Text :
    def __init__(self):
        self.char_to_int = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35}
        self.int_to_char = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'}
        self.max_len = 6
        self.model = CNNRNNModel(num_classes=len(self.char_to_int),max_len=self.max_len)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('captcha_model.pth', map_location=self.device))
        self.model.eval()

    def FindText(self,img) :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (300, 75))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        text = ''
        with torch.no_grad():
            self.model.eval()
            img_tensor = torch.tensor(img).unsqueeze(0).to(self.device)
            outputs = self.model(img_tensor)
            outputs = outputs.permute(1, 0, 2)
            _, preds = torch.max(outputs, 2)
            preds = preds.view(-1)
            for i in range(6):
                text += self.int_to_char[preds[i].item()]

        return text
    
if __name__ == '__main__':
    img2text = Image2Text()
    img = cv2.imread('test.jpg')
    text = img2text.FindText(img)
    print(text)