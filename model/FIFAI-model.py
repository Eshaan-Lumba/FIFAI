# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from argparse import ArgumentParser
from fastprogress.fastprogress import master_bar, progress_bar

import torch
from torch import nn

import pandas as pd
import os
import numpy as np


# -

def train_one_epoch(dataloader, model, criterion, optimizer, device, mb):

    # Put the model into training mode
    model.train()

    # Loop over the data using the progress_bar utility
    for _, (X, Y) in progress_bar(DataLoaderProgress(dataloader), parent=mb):
        X, Y = X.to(device), Y.to(device)

        # Compute model output and then loss
        output = model(X)
        loss = criterion(output, Y)

        # - zero-out gradients
        optimizer.zero_grad()
        # - compute new gradients
        loss.backward()
        # - update paramters
        optimizer.step()


def validate(dataloader, model, criterion, device, epoch, num_epochs, mb):

    # Put the model into validation/evaluation mode
    model.eval()

    N = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    loss, num_correct = 0, 0

    # Tell pytorch to stop updating gradients when executing the following
    with torch.no_grad():

        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            # Compute the model output
            output = model(X)
            
            print(X)
            print(Y)
            print(output)

            # - compute loss
            loss += criterion(torch.flatten(output), Y).item()
            # - compute the number of correctly classified examples
            num_correct += (output.argmax(1) == Y).type(torch.float).sum().item()

        loss /= num_batches
        accuracy = num_correct / N

    message = "Initial" if epoch == 0 else f"Epoch {epoch:>2}/{num_epochs}:"
    message += f" accuracy={100*accuracy:5.2f}%"
    message += f" and loss={loss:.3f}"
    mb.write(message)


def train(model, criterion, optimizer, train_loader, valid_loader, device, num_epochs):
    mb = master_bar(range(num_epochs))

    validate(valid_loader, model, criterion, device, 0, num_epochs, mb)

    for epoch in mb:
        train_one_epoch(train_loader, model, criterion, optimizer, device, mb)
        validate(valid_loader, model, criterion, device, epoch + 1, num_epochs, mb)


# +
# creates a model for each team file in the 'teams_five_season_data' folder
def load_data_into_models(path): 
    team_models = []

    for file in os.listdir(path):
        model = Team(path+"/"+file)
        team_models.append(model)
    
    return team_models

# creates dataloaders from each team model
def load_team_models_into_dls(team_models):
    team_dataloaders = []
    for team in team_models:
        dl = torch.utils.data.DataLoader(team, batch_size = 10)
        team_dataloaders.append(dl)
        
    return team_dataloaders


# -

# converts team names into readable integer encodings
# might be simpler just to assign each team an integer
def encode_string_as_int(string):
    ret = ""
    for c in string:
        ret += str(ord(c))
    return int(ret)


# class to create individual model objects for each team
class Team ():
    def __init__(self, file_path):
        file_out = pd.read_csv(file_path)
        # x_axis_labels = file_out.iloc[0, 1:8].values
        
        data = file_out.iloc[1:191, 1:8].values
        x = []
        y = []
        
        for match in data:
            # convert team names to int encoding
            match[0] = encode_string_as_int(match[0])
            match[1] = encode_string_as_int(match[1])
            
            # create lists for input and target data
            x.append(np.delete(match, 4).tolist())
            y.append(match[4])
            
        # cast all string input data to ints
        for match in x:
            for attr in range(0,6):
                match[attr] = int(match[attr])
                
        # cast all string target data to ints 
        for match_result in range(len(y)):
            y[match_result] = int(y[match_result])
    
        
        # print("training data:")
        # print(x)
        # print()
        
        # print("match results (aka targets):")
        # print(y)
        
        # final data to train on
        self.X_train = torch.tensor(x, dtype = torch.float).float()
        # targets (i.e. match scorelines)
        self.Y_train = torch.tensor(y, dtype= torch.long)
        
        # print (self.X_train)
        # print (self.Y_train)
    
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


def main():
    aparser = ArgumentParser("FIFAI--Train a neural network to predict EPL scorelines.")
    aparser.add_argument("epl_data", type=str, help="Path to store/find the EPL games dataset")
    aparser.add_argument("--num_epochs", type=int, default=10)
    aparser.add_argument("--batch_size", type=int, default=128)
    aparser.add_argument("--learning_rate", type=float, default=0.01)
    aparser.add_argument("--gpu", action="store_true")

    args = aparser.parse_args()

    # Use GPU if requested and available
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    
    # Get models
    team_models = load_data_into_models(args.epl_data)

    # Get dataloaders
    dls = load_team_models_into_dls(team_models)
    
    # Using the Arsenal model, for example
    train_loader = dls[0]
    valid_loader = dls[0]
    
    model = torch.nn.Sequential(nn.Flatten(), torch.nn.Linear(in_features=6, out_features=1))
    
    # - specifies CrossEntropyLoss as loss criterion
    # - specifies Adam as our optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, criterion, optimizer, train_loader, valid_loader, device, args.num_epochs)


# !python FIFAI-model.py "../teams_five_season_data"

# +
# class RNN_Model(nn.Module):
#     def __init__(self, input_size, output_size, hidden_dim, n_layers):
#         super(Model, self).__init__()

#         # Defining some parameters
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers

#         # RNN layer
#         self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
#         # Linear layer
#         self.linear = nn.Linear(hidden_dim, output_size)
    
#     def forward(self, x):
#         batch_size = x.size(0)

#         # Initializing hidden state for first input using method defined below
#         hidden = self.init_hidden(batch_size)

#         # Passing in the input and hidden state into the model and obtaining outputs
#         out, hidden = self.rnn(x, hidden)
        
#         # Reshaping the outputs such that it can be fit into the fully connected layer
#         out = out.contiguous().view(-1, self.hidden_dim)
#         out = self.linear(out)
        
#         return out, hidden
    
#     def init_hidden(self, batch_size):
#         # This method generates the first hidden state of zeros which we'll use in the forward pass
#         # We'll send the tensor holding the hidden state to the device we specified earlier as well
#         hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
#         return hidden
# -

if __name__ == "__main__":
    main()
