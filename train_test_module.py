import torch
import torch.nn as nn
import numpy as np
import data
from torch.utils.data import Dataset, DataLoader, random_split
from nnmodels import CNN, SimpleRNN, SimpleLSTM
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import plot_functions
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold



class MyDataset(Dataset):
    def __init__(self, data, labels, groups = None, exclude_subject=None,  only_subject=None):
        self.data = data
        self.labels = (labels > 1) * 1
        #self.labels = self.labels.to(torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.groups = groups

        # For the model pr subject groups is not needed
        if groups is not None:
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


def cv_logo(model_class, data, labels, groups, num_epochs=10, batch_size=16):
    """
    Cross validation loop that performs leave one group (subject) out 
    Input the wanted model class
    """
    unique_subjects = get_unique_subjects(groups)
    num_subjects = len(unique_subjects)
    
    # Store the validation results for each fold
    validation_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    all_y_true = []
    all_y_pred = []

    # Iterate through the subjects, leaving one subject out at a time
    for i in range(num_subjects):
        exclude_subject = unique_subjects[i]
        print("excluded subject nr: ", exclude_subject)

        # Create the train and validation datasets
        train_dataset = MyDataset(data, labels, groups, exclude_subject = exclude_subject)
        val_dataset = MyDataset(data, labels, groups, only_subject=exclude_subject)

        # Create the DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the model and optimizer
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        criterion = nn.BCELoss()

        # Train and evaluate the model using train_loader and val_loader
        # For each epoch, update the model and calculate validation performance
        for epoch in range(num_epochs):
            # Train the model
            model.train()

            ncorrect_train = 0
            for batch_idx, (data_batch, label_batch) in enumerate(train_loader):
                # Forward pass
                # batchsize på 16 med en dataloader
                y_pred = model(data_batch)
                y_pred = y_pred.squeeze()
                y_pred_categorical = y_pred > 0.5
                ncorrect_train += torch.sum(label_batch == y_pred_categorical).item()
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

            eps = 1e-11
            precision = true_positives / (true_positives + false_positives + eps)
            recall = true_positives / (true_positives + false_negatives) + eps
            f1 = 2 * (precision * recall) / (precision + recall + eps)

            precision_results.append(precision)
            recall_results.append(recall)
            f1_results.append(f1)

            print(f"Fold {i + 1}, validation precision: {precision}, validation recall: {recall}, validation F1-score: {f1}")

            # Accumulate true labels and predicted labels for the overall confusion matrix
            y_true = val_dataset.labels.numpy()
            y_pred = [1 if p > 0.5 else 0 for p in model(val_dataset.data).squeeze().detach().numpy()]
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            ##confusion matrix pr subject
            #if confmatprsub == True: 
            #    y_true = val_dataset.labels.numpy()
            #    y_pred = [1 if p > 0.5 else 0 for p in model(val_dataset.data).squeeze().detach().numpy()]
            #    plot_functions.confmat(y_true, y_pred, f"Fold {i + 1}, validation accuracy {validation_accuracy}")

        # Store the validation results for this fold
        validation_results.append(ncorrect/len(val_dataset))

    # Calculate the average validation result across all folds
    avg_validation_result = np.mean(validation_results)

    # Plot the overall confusion matrix after the cross-validation loop
    plot_functions.confmat(np.array(all_y_true), np.array(all_y_pred), f"Overall validation accuracy {avg_validation_result*100:.2f}%")

    return avg_validation_result, validation_results, precision_results, recall_results, f1_results


def train_model_per_subject(model_class, X_train, y_train, X_test, y_test, num_epochs=10, batch_size=16):
    """
    Trains an individual model of all subject's data.
    Used in kfold_cross_validation_per_subject.
    """
    # Create the DataLoaders
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model and optimizer
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.BCELoss()

    # Train and evaluate the model using train_loader and test_loader
    for epoch in range(num_epochs):
        # Train the model
        model.train()

        ncorrect_train = 0
        for batch_idx, (data_batch, label_batch) in enumerate(train_loader):
            # Forward pass
            y_pred = model(data_batch)
            y_pred = y_pred.squeeze()
            y_pred_categorical = y_pred > 0.5
            ncorrect_train += torch.sum(label_batch == y_pred_categorical).item()

            loss = criterion(y_pred.view(-1), label_batch.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch}, train accuracy {ncorrect_train/len(train_dataset)}")

        # Evaluate the model on the test set
        model.eval()
        ncorrect = 0
        ntotal = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for batch_idx, (data_batch, label_batch) in enumerate(test_loader):
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

    eps = 1e-11
    precision = true_positives / (true_positives + false_positives + eps)
    recall = true_positives / (true_positives + false_negatives) + eps
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return validation_accuracy, precision, recall, f1




def kfold_prsubject_stratified(X, y, groups, model_class, num_epochs=10, batch_size=16):
    """
    Models trained individually pr subject.
    Using stratified 10-fold cross validation
    Returns the accuracy for the 10 folds, and the mean score
    """
    y = np.array(y)
    groups = np.array(groups)

    tot_scores = []
    tot_indi_scores = []
    for subject in set(groups):
        print(f"Training on subject {subject}")
        scores = []
        temp_subject = np.where(groups == subject)
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold = 0
        for train_index, test_index in skf.split(X[temp_subject[0]], y[temp_subject[0]]):
            print(f"fold no. {fold}")
            X_train, X_test = X[temp_subject[0]][train_index], X[temp_subject[0]][test_index]
            y_train, y_test = y[temp_subject[0]][train_index], y[temp_subject[0]][test_index]
            accuracy, precision, recall, f1 = train_model_per_subject(model_class, X_train, y_train, X_test, y_test, num_epochs, batch_size)
            scores.append(accuracy)
            fold += 1

        tot_indi_scores.append(scores)
        tot_scores.extend(scores)

    mean_indi_scores = [np.mean(i) for i in tot_indi_scores]

    return tot_scores, tot_indi_scores, mean_indi_scores



if __name__ == "__main__":
    X,y,groups = data.datapreprocess_tensor_cnn()
    my_model = CNN

    # Perform cross-validation logo
    avg_validation_result, validation_results, precision_results, recall_results, f1_results = cv_logo(my_model, data = X, labels = y, groups = groups, num_epochs=10, batch_size=16)
    #plot_functions.barplot(groups, validation_results, acc = avg_validation_result, xtype_title = "X")

    #print("Average validation result:", avg_validation_result)
    #print("Average precision result:", precision_results)
    #print("Average recall result:", recall_results)
    #print("Average validation result:", f1_results)

    # Perform cross-validation pr subject
    #tot_scores, tot_indi_scores, mean_indi_scores = kfold_prsubject_stratified(X, y, groups, my_model, num_epochs=10, batch_size=16)
    #plot_functions.barplot(groups, mean_indi_scores, acc = np.mean(tot_scores))
