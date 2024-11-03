# **Common Neural Network Architectures Cheat Sheet**

---

## **1. Vanilla Neural Network (Fully Connected Network)**

A simple feedforward neural network that consists of fully connected layers. Typically used for tabular data.

```python
import torch
import torch.nn as nn

class VanillaNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size) # Output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = VanillaNN(input_size=10, hidden_size=64, output_size=2)
```

### **Notes**
- **`input_size`**: Number of input features.
- **`hidden_size`**: Size of each hidden layer.
- **Activation**: ReLU is commonly used for hidden layers in vanilla neural networks.

---

## **2. Convolutional Neural Network (CNN)**

A CNN is often used for image data and consists of convolutional, pooling, and fully connected layers.

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First conv layer
        x = self.pool(torch.relu(self.conv2(x)))  # Second conv layer
        x = x.view(-1, 32 * 8 * 8)               # Flatten
        x = torch.relu(self.fc1(x))              # Fully connected layer
        x = self.fc2(x)                          # Output layer
        return x

# Instantiate the model
model = SimpleCNN(num_classes=10)
```

### **Notes**
- **Convolutional Layers**: `in_channels` for RGB images is typically 3, `out_channels` increases with layers.
- **Pooling Layers**: Often used to downsample, typically MaxPooling with kernel size (2,2).
- **Flattening**: Converts the 2D feature maps into 1D before the fully connected layers.

---

## **3. Recurrent Neural Network (RNN)**

An RNN for sequential data, such as time series or text, where each time step depends on the previous time steps.

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)   # RNN layer output
        out = self.fc(out[:, -1, :])  # Fully connected layer on last time step
        return out

# Instantiate the model
model = SimpleRNN(input_size=128, hidden_size=64, output_size=10)
```

### **Notes**
- **`input_size`**: Size of each input element (e.g., embedding size for text).
- **`hidden_size`**: Number of features in the hidden state.
- **Output**: Only the last time step’s output is usually passed to the fully connected layer for classification.

---

## **4. Long Short-Term Memory (LSTM)**

An LSTM network, a type of RNN with gates that help capture long-term dependencies.

```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Fully connected layer on last time step
        return out

# Instantiate the model
model = SimpleLSTM(input_size=128, hidden_size=64, output_size=10)
```

### **Notes**
- LSTM parameters are similar to RNNs but include gates for better handling of sequential dependencies.
- Suitable for NLP, speech, and time series tasks.

---

## **5. Transformer Network**

The Transformer is a model for sequential data, leveraging self-attention rather than recurrence.

```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, hidden_dim, output_size):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads), num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(input_dim, output_size)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x.mean(dim=1))  # Global average pooling
        return x

# Instantiate the model
model = TransformerModel(input_dim=512, num_heads=8, num_encoder_layers=6, hidden_dim=2048, output_size=10)
```

### **Notes**
- **`d_model`**: Input size, often equal to embedding size.
- **Self-Attention**: The number of heads in multi-head attention; typically 8 for smaller tasks.
- Used for text and sequential data; excels in NLP tasks.

---

## **6. Autoencoder**

An autoencoder is an unsupervised network used for dimensionality reduction or denoising.

```python
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # For normalized data
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = Autoencoder(input_size=784, hidden_size=128)
```

### **Notes**
- **Encoder**: Compresses input to a lower-dimensional representation.
- **Decoder**: Reconstructs the input from the compressed representation.
- Common in image compression, denoising, and anomaly detection.

---

## **7. Generative Adversarial Network (GAN)**

A GAN consists of two networks: a generator that creates synthetic data and a discriminator that differentiates real vs. fake data.

```python
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        return self.fc(x)

# Instantiate models
noise_dim = 100
input_dim = 784  # For 28x28 images flattened
generator = Generator(noise_dim, input_dim)
discriminator = Discriminator(input_dim)
```

### **Notes**
- **Generator**: Takes random noise as input and generates data.
- **Discriminator**: Predicts real or fake labels for input data.
- Commonly used in image generation, data augmentation, and creative applications.
