# **Comprehensive PyTorch Cheat Sheet**

---

## **1. Basic Setup and Configuration**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Set device (CPU/GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## **2. Tensors**
### **Creating Tensors**
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # 1D tensor
zeros = torch.zeros((3, 3))       # 3x3 tensor of zeros
ones = torch.ones((2, 2))         # 2x2 tensor of ones
randn = torch.randn((2, 3))       # 2x3 tensor of standard normal
identity = torch.eye(4)           # 4x4 identity matrix
arange = torch.arange(0, 10, 1)   # Tensor with values from 0 to 9
linspace = torch.linspace(0, 1, 10)  # 10 values from 0 to 1

# Move tensor to GPU
x = x.to(device)
```

### **Tensor Operations**
```python
# Reshaping
x_reshaped = x.view(1, -1)        # Reshape to a 1xN tensor
y = x.view(-1, 1)                 # Reshape to Nx1

# Arithmetic Operations
add = x + y                       # Element-wise addition
mul = x * y                       # Element-wise multiplication
dot_product = torch.dot(x, y)     # Dot product

# Reduction Operations
mean = x.mean()                   # Mean of tensor
sum = x.sum()                     # Sum of tensor
max_val, max_idx = x.max(dim=1)   # Max values and indices along dimension

# Matrix Multiplication
matmul = torch.mm(x, y)           # Matrix multiplication
```

---

## **3. Automatic Differentiation with Autograd**
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2 + 3                    # Some operations
y.sum().backward()                # Backpropagate to compute gradients
print(x.grad)                     # Access gradients

# Disable gradient tracking (for inference)
with torch.no_grad():
    y = x * 2
```

---

## **4. Neural Network Layers and Building a Model**

### **Common Layers**
```python
# Fully Connected Layer
fc = nn.Linear(in_features=128, out_features=64)

# Convolutional Layer
conv2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

# Pooling Layer
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Dropout Layer
dropout = nn.Dropout(p=0.5)

# Batch Normalization
batchnorm = nn.BatchNorm1d(num_features=128)
```

### **Recurrent Layers**
``` python
# RNN Layer
rnn = nn.RNN(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

# LSTM Layer
lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

# GRU Layer
gru = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
```

### **Defining a Model**
```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # Convolution + ReLU
        x = self.pool(x)                # Pooling
        x = x.view(-1, 32 * 14 * 14)    # Flatten
        x = F.relu(self.fc1(x))         # Fully Connected + ReLU
        x = self.fc2(x)                 # Output layer
        return x

# Instantiate the model and move to GPU
model = MyModel().to(device)
```

---

## **5. Loss Functions**
```python
# Classification Loss
criterion = nn.CrossEntropyLoss()         # Cross-entropy for classification

# Regression Losses
mse_loss = nn.MSELoss()                   # Mean Squared Error
mae_loss = nn.L1Loss()                    # Mean Absolute Error

# Custom Loss (Example)
class CustomLoss(nn.Module):
    def forward(self, output, target):
        return torch.mean((output - target) ** 2)
```

---

## **6. Optimizers**
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = optim.Adam(model.parameters(), lr=0.001)
rmsprop = optim.RMSprop(model.parameters(), lr=0.01)
```

---

## **7. Training Loop Template**
```python
epochs = 10
for epoch in range(epochs):
    model.train()                        # Set model to training mode
    running_loss = 0.0
    for data, labels in DataLoader(train_dataset, batch_size=32, shuffle=True):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()            # Zero gradients
        outputs = model(data)            # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                  # Backpropagation
        optimizer.step()                 # Update weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataset)}")
```

---

## **8. Evaluation and Metrics**
```python
# Evaluation loop
model.eval()                              # Set to eval mode
correct = 0
total = 0
with torch.no_grad():
    for data, labels in DataLoader(test_dataset, batch_size=32):
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy * 100}%')
```

---

## **9. Save and Load Models**
```python
# Save model state
torch.save(model.state_dict(), 'model.pth')

# Load model state
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.to(device)
```

---

## **10. Custom Datasets and DataLoader**
```python
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Load custom dataset
train_data = CustomDataset(train_images, train_labels)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
```

---

## **11. Common Tricks & Tips**

### **Gradient Clipping**
```python
# Clip gradients to avoid exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### **Learning Rate Scheduling**
```python
# Adjust learning rate dynamically
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler.step()  # Call this at the end of each epoch
```

### **Freezing Layers**
```python
# Freeze all parameters except for last layer
for param in model.parameters():
    param.requires_grad = False
model.fc2.requires_grad = True
```

### **Mixed Precision Training (for GPUs)**
```python
scaler = torch.cuda.amp.GradScaler()     # Initialize scaler for mixed precision

# Training loop with mixed precision
for data, labels in train_loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = model(data)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
