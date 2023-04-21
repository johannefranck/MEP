import torch
import torch.nn as nn
import numpy as np
import data
from torch.utils.data import Dataset, DataLoader, random_split
from nnmodels import CNN, SimpleRNN, SimpleLSTM



# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = (labels > 1) * 1
        self.labels = self.labels.to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
    

# Instantiate the CNN model
#model = CNN()
# Hyperparameters
input_size = 1
hidden_size = 128
output_size = 1
num_layers = 1
seq_length = 85

# Create the model
model = SimpleLSTM(input_size, hidden_size, output_size, num_layers)


# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load and preprocess the data
main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
filelist = data.get_all_paths(main_path)
X, y, groups, list_subjects = data.get_all_data(filelist)
X = np.transpose(X) 
X = torch.from_numpy(X).float()
X = X.unsqueeze(1) # Add a channel dimension # (number of signals, number of input channels, signal length)
y = torch.tensor(y).float()




dataset = MyDataset(X, y)
# Create the DataLoader
batch_size = 16
shuffle = True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Iterate through the DataLoader
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

# Train the model
for epoch in range(30):
    ncorrect_train = 0
    for batch_idx, (data_batch, label_batch) in enumerate(train_loader):
        """print(f"Batch {batch_idx + 1}:")
        print(f"Data batch shape: {data_batch.shape}")
        print(f"Label batch shape: {label_batch.shape}")"""
    # Forward pass
    #badgesize på 16 med en dataloader 
        y_pred = model(data_batch)
        y_pred = y_pred.squeeze()
        y_pred_categorical = y_pred > 0.5
        ncorrect_train += torch.sum(label_batch == y_pred_categorical).item()

        loss = criterion(y_pred, label_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch_idx % 100 == 0:
        #    print("loss" + str(loss.item()))
            
    # Print the loss every 5 epochs
    if epoch % 1 == 0:
        
        print(f"epoch {epoch}, train accuracy {ncorrect_train/train_size}")
        #print(f"Epoch {epoch}, Loss {loss.item()}")
        model.eval()
        ncorrect = 0
        for batch_idx, (data_batch, label_batch) in enumerate(val_loader):
            
            y_pred = model(data_batch)
            y_pred = y_pred.squeeze()
            y_pred_categorical = y_pred > 0.5
            ncorrect += torch.sum(label_batch == y_pred_categorical).item()

        print(f"epoch: {epoch}, validation accuracy {ncorrect/val_size}")
        model.train()

