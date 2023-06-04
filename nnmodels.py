import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)     
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        x = x.permute(0,2,1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)# lstm
        out, _ = self.lstm(x, h0) # lstm
        #out, _ = self.gru(x, h0)
        #out, _ = self.rnn(x, h0)
        # Extract the output from the last time step
        out = out[:, -1, ]

        # Pass the output through a fully connected layer
        out = self.fc(out)
        out = nn.functional.sigmoid(out)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=64*10, out_features=128, bias = True) #trying with bias 64*10
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
    def forward(self, x):
        batch_size = x.size(0)
        #print("Input shape: ", x.shape)
        # Convolutional layers
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        #print("After Conv1 and Pool1 shape: ", x.shape)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        #print("After Conv2 and Pool2 shape: ", x.shape)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool3(x)
        #print("After Conv3 and Pool3 shape: ", x.shape)

        # Flatten and fully connected layers
        x = torch.flatten(x, start_dim=1)
        #print("After Flattening shape: ", x.shape)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)


        # Apply sigmoid activation function to the output
        #x = torch.sigmoid(x)
        x = torch.sigmoid(x).view(-1)

        return x
    
#original CNN model. baseline model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(16 * 42, 1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x).view(-1)
        return x


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * num_directions, output_size)

    def forward(self, x):
        x = x.permute(0,2,1)
        # Initialize hidden state and cell state
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Extract the output from the last time step
        out = out[:, -1,:]

        # Pass the output through a fully connected layer
        out = self.fc(out)
        out = nn.functional.sigmoid(out)
        return out

"""class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(16 * 42, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)  # use dropout here
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x).view(-1)
        return x"""
"""# Hyperparameters
vocab_size = 10000
embed_size = 256
num_blocks = 4

# Instantiate the model and optimizer
model = SimpleTransformer(vocab_size, embed_size, num_blocks)"""
if __name__ == "__main__":
    # Hyperparameters
    input_size = 10
    hidden_size = 128
    output_size = 2
    num_layers = 1
    seq_length = 5

    # Create the model
    model = SimpleRNN(input_size, hidden_size, output_size, num_layers)