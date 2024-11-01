import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

X_train = np.arange(10, dtype='float32').reshape((10,1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype="float32")

X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train) # normalize by mean centering and deviding by standard deviation
X_train_norm = torch.from_numpy(X_train_norm) # convert to tensor

y_train = torch.from_numpy(y_train).float()
train_ds = TensorDataset(X_train_norm, y_train)

batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

def loss_fn(input, target):
    return (input - target).pow(2).mean()



# using the build in functions instead (manual implmentation in torch_linear.py)
loss_fn = nn.MSELoss(reduction='mean')

num_epochs = 200
log_epochs = 10

input_size = 1
output_size = 1
learning_rate = 0.001

model = nn.Linear(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for X_batch, y_batch in train_dl:
        # prediction
        pred = model(X_batch)[:, 0]
        # Calculate loss
        loss = loss_fn(pred, y_batch)
        # Compute gradients
        loss.backward()
        # update parameters using the gradients
        optimizer.step()
        #  reset the gradients
        optimizer.zero_grad()
    if epoch % log_epochs==0:
        print(f"Epoch: {epoch} | loss: {loss.item():.4f}")


# plot the final model
print(f"Final Parameters: Weight: {model.weight.item()} | Bias: {model.bias.item()}")
