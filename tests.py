import matplotlib.pyplot as plt
import numpy as np
import data
import models


import torch
import torch.nn as nn
import torcheeg
from torcheeg import models
import torcheeg.models

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(SimpleTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)
        self.output_proj = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        
        batch_self_attention_weights = []
        for batch_idx in range(batch_size):
            signal_x = x[batch_idx].unsqueeze(0)
            signal_self_attention_weights = []
            for i in range(len(self.transformer.layers)):
                signal_x, self_attn_weights = self.transformer.layers[i].self_attn(signal_x, signal_x, signal_x)
                signal_self_attention_weights.append(self_attn_weights.detach())
            batch_self_attention_weights.append(signal_self_attention_weights)
            x[batch_idx] = signal_x.squeeze(0)
        
        x = x[:, -1, :]  # Get the last token's output
        x = self.output_proj(x)
        x = self.sigmoid(x)
        return x, batch_self_attention_weights
    

def plot_attention_heatmap(batch_attention_weights, signal, layer, head):
    weights = batch_attention_weights[signal][layer][head].cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.title(f'Signal {signal + 1}, Layer {layer + 1}, Head {head + 1}')
    plt.xlabel('Target tokens')
    plt.ylabel('Source tokens')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess the data
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = data.get_all_paths(main_path)
    X, y, groups, list_subjects = data.get_all_data(filelist)
    num_signals = X.shape[1]
    X = np.reshape(X, (num_signals, 85, 1))

    X = torch.from_numpy(X).float()
    #X = X.unsqueeze(1) # Add a channel dimension # (number of signals, number of input channels, signal length)
    y = torch.tensor(y).float()
    # Replace these variables with your actual data and model
    data = X
    subset_data = data[:5]
    labels = y
    groups_data = groups
    #print data size and subset data size
    print("data size:", data.shape)
    print("subset data size:", subset_data.shape)

    # Hyperparameters
    input_dim = 1
    d_model = 128
    nhead = 4
    num_layers = 2

    # Create a random input signal of size (batch_size, seq_length, input_dim)
    input_signal = subset_data
    # Initialize the SimpleTransformer model
    model = SimpleTransformer(input_dim, d_model, nhead, num_layers)

    # Test the transformer on the input signal
    output_probs, batch_self_attention_weights = model(input_signal)
    print("Output probabilities shape:", output_probs.shape)
    print("Output probabilities:", output_probs)


    #plot_attention_heatmap(batch_self_attention_weights, signal=0, layer=0, head=0)

    
