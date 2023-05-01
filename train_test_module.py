import torch
import torch.nn as nn
import numpy as np
import data
from torch.utils.data import Dataset, DataLoader, random_split
from nnmodels import CNN, SimpleRNN, SimpleLSTM
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


# Custom Dataset
#dataset without grouping
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
    


# Load and preprocess the data
main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
filelist = data.get_all_paths(main_path)
X, y, groups, list_subjects = data.get_all_data(filelist)
X_norm = data.normalize_X(X, groups)
X = np.transpose(X) 
X = torch.from_numpy(X).float()
X = X.unsqueeze(1) # Add a channel dimension # (number of signals, number of input channels, signal length)
y = torch.tensor(y).float()


#Hyperparameters
num_epochs = 50
batch_size = 16
learning_rate = 0.0001
weight_decay= 1e-5

#scheduler = StepLR(optimizer, step_size=10, gamma=0.1) #lowering lr while training

dataset = MyDataset(X, y)
# Create the DataLoader
shuffle = True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



    

# Instantiate the CNN model
model = CNN()
"""# Hyperparameters for specific model
input_size = 1
hidden_size = 128
output_size = 1
num_layers = 1
seq_length = 85

# Create the model
model = SimpleLSTM(input_size, hidden_size, output_size, num_layers)
"""

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)





# Iterate through the DataLoader
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

accuracies_train = []
accuracies_val = []
# Train the model
for epoch in range(num_epochs):
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
    
    #running validation
    print(f"epoch {epoch}, train accuracy {ncorrect_train/train_size}")
    model.eval()
    ncorrect = 0
    for batch_idx, (data_batch, label_batch) in enumerate(val_loader):
        
        y_pred = model(data_batch)
        y_pred = y_pred.squeeze()
        y_pred_categorical = y_pred > 0.5
        ncorrect += torch.sum(label_batch == y_pred_categorical).item()


    print(f"epoch: {epoch}, validation accuracy {ncorrect/val_size}")

    #Setting back to train mode
    model.train()

    # Update the learning rate using the scheduler
    #scheduler.step()
    #print(f"Epoch {epoch + 1}, Learning rate: {scheduler.get_last_lr()[0]}")


    accuracies_train.append(ncorrect_train/train_size)
    accuracies_val.append(ncorrect/val_size)

print(f"mean of validation accuracy: {np.mean(accuracies_val)}")
plt.title(f"CNN(X): num_e:{num_epochs}, bs:{batch_size}, lr:{learning_rate}, wd:{weight_decay}")
plt.plot(accuracies_train, color = "red")
plt.plot(accuracies_val, color = "green")
plt.show()

