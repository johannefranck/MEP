import torch
import torch.nn as nn
import numpy as np
import data
from torch.utils.data import Dataset, DataLoader, random_split
from nnmodels import CNN, SimpleRNN, SimpleLSTM, SimpleTransformer
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import plot_functions


class MyDataset(Dataset):
    def __init__(self, data, labels, groups, exclude_subject=None,  only_subject=None):
        self.data = data
        self.labels = (labels > 1) * 1
        self.labels = self.labels.to(torch.float32)
        self.groups = groups

        # Exclude the subject if specified
        if exclude_subject is not None:
            mask = np.array(groups) != exclude_subject
            self.data = self.data[mask]
            self.labels = self.labels[mask]
        elif only_subject is not None:
            mask = np.array(groups) == only_subject
            self.data = self.data[mask]
            self.labels = self.labels[mask]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label

def get_unique_subjects(groups):
    return list(set(groups))


def cross_validate(model_class, data, labels, groups, num_epochs=10, batch_size=16):
    unique_subjects = get_unique_subjects(groups)
    num_subjects = len(unique_subjects)
    
    # Store the validation results for each fold
    validation_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    
    # Iterate through the subjects, leaving one out at a time
    for i in range(num_subjects):
        exclude_subject = unique_subjects[i]
        print("excluded subject nr: ", exclude_subject)

        # Create the train and validation datasets
        train_dataset = MyDataset(data, labels, groups, exclude_subject = exclude_subject)
        val_dataset = MyDataset(data, labels, groups, only_subject=exclude_subject)



        # Create the DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        #hyperparameters
        vocab_size = 84
        embed_size = 128
        num_blocks = 6

        # Initialize the model and optimizer
        model = model_class(vocab_size, embed_size, num_blocks)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        criterion = nn.BCELoss()

        # Train and evaluate the model using train_loader and val_loader
        # For each epoch, update the model and calculate validation performance
        # You can use your existing training and evaluation code here
        for epoch in range(num_epochs):
            # Train the model
            model.train()

            ncorrect_train = 0
            for batch_idx, (data_batch, label_batch) in enumerate(train_loader):
                """print(f"Batch {batch_idx + 1}:")
                print(f"Data batch shape: {data_batch.shape}")
                print(f"Label batch shape: {label_batch.shape}")"""
                #Forward pass
                #badgesize på 16 med en dataloader
                y_pred = model(data_batch)
                #y_pred = model.__call__(data_batch)
                y_pred = y_pred.squeeze()
                y_pred_categorical = y_pred > 0.5
                ncorrect_train += torch.sum(label_batch == y_pred_categorical).item()

                #loss = criterion(y_pred, label_batch)
                loss = criterion(y_pred.view(-1), label_batch.view(-1))


                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"epoch {epoch}, train accuracy {ncorrect_train/len(train_dataset)}")
            # Evaluate the model on the validation set
            model.eval()
            ncorrect = 0
            ntotal = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            for batch_idx, (data_batch, label_batch) in enumerate(val_loader):
                #print(f"Data batch shape: {data_batch.shape}")

                y_pred = model(data_batch)
                y_pred = y_pred.squeeze()
                y_pred_categorical = y_pred > 0.5
                ncorrect += torch.sum(label_batch == y_pred_categorical).item()
                ntotal += label_batch.size(0)
                true_positives += torch.sum((label_batch == 1) & (y_pred_categorical == 1)).item()
                false_positives += torch.sum((label_batch == 0) & (y_pred_categorical == 1)).item()
                false_negatives += torch.sum((label_batch == 1) & (y_pred_categorical == 0)).item()

            validation_accuracy = ncorrect / ntotal
            print(f"epoch: {epoch}, validation accuracy {validation_accuracy}")

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = 2 * (precision * recall) / (precision + recall)

            precision_results.append(precision)
            recall_results.append(recall)
            f1_results.append(f1)

            print(f"Fold {i + 1}, validation precision: {precision}, validation recall: {recall}, validation F1-score: {f1}")


        # Store the validation results for this fold
        validation_results.append(ncorrect/len(val_dataset))

    # Calculate the average validation result across all folds
    avg_validation_result = np.mean(validation_results)

    return avg_validation_result, validation_results


if __name__ == "__main__":
    # Load and preprocess the data
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = data.get_all_paths(main_path)
    X, y, groups, list_subjects = data.get_all_data(filelist)
    X_norm = data.normalize_X(X, groups)
    X = np.transpose(X)
    X = torch.from_numpy(X).float()
    X = X.unsqueeze(1) # Add a channel dimension # (number of signals, number of input channels, signal length)
    y = torch.tensor(y).float()
    # Replace these variables with your actual data and model
    data = X
    labels = y
    groups_data = groups
    my_model = SimpleTransformer

    # Perform cross-validation
    avg_validation_result, validation_results = cross_validate(my_model, data, labels, groups_data, num_epochs=1, batch_size=16)

    plot_functions.barplot(groups, validation_results, acc = avg_validation_result, xtype_title = "X")

    print("Average validation result:", avg_validation_result)
