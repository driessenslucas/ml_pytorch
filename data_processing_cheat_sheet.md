# **Data Preprocessing and Loading Cheat Sheet**

---

## **1. Data Transformations**
PyTorch uses the `torchvision.transforms` module to apply standard transformations, particularly for image data. Here’s a look at common transformations:

```python
from torchvision import transforms

# Basic Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),         # Resize to fixed size
    transforms.RandomHorizontalFlip(),     # Random horizontal flip
    transforms.RandomRotation(10),         # Random rotation within 10 degrees
    transforms.ToTensor(),                 # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (example for grayscale)
])
```

### **Common Transformation Functions**
- **transforms.Resize((H, W))**: Resizes images to the given height \( H \) and width \( W \).
- **transforms.CenterCrop((H, W))**: Crops the center of the image to size \( H \times W \).
- **transforms.RandomRotation(degrees)**: Rotates the image by a random degree.
- **transforms.RandomResizedCrop(size)**: Crops a random part of the image to a specified size.
- **transforms.ColorJitter**: Randomly changes brightness, contrast, saturation, and hue.
- **transforms.Normalize(mean, std)**: Normalizes the tensor by mean and standard deviation.

### **Custom Transformations**
You can define custom transformations by creating a class that implements `__call__`:

```python
class CustomTransform:
    def __call__(self, x):
        # Perform custom transformations
        return x * 2
```

---

## **2. Custom Dataset**
When working with your own datasets, define a custom `Dataset` by subclassing `torch.utils.data.Dataset`.

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# Instantiate dataset
dataset = CustomDataset(data=my_data, labels=my_labels, transform=transform)
```

### **Key Components**
- **`__len__`**: Returns the size of the dataset.
- **`__getitem__`**: Retrieves a sample and applies transformations, if any.

---

## **3. DataLoader**

The `DataLoader` allows you to load data in batches, shuffle, and parallelize loading with multiple workers.

```python
from torch.utils.data import DataLoader

# Instantiate DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
```

### **DataLoader Parameters**
- **`batch_size`**: Number of samples per batch.
- **`shuffle`**: Whether to shuffle the data every epoch.
- **`num_workers`**: Number of subprocesses for data loading (useful for large datasets).
- **`pin_memory`**: Speeds up data transfer to CUDA-enabled GPUs by using pinned memory.

---

## **4. Working with Image Datasets (torchvision.datasets)**

PyTorch’s `torchvision.datasets` provides built-in support for many popular image datasets.

```python
from torchvision import datasets

# Load CIFAR-10 with transformations
train_dataset = datasets.CIFAR10(root='data/', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='data/', train=False, transform=transform, download=True)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
```

### **Other Built-in Datasets**
- **MNIST**: `datasets.MNIST(...)`
- **FashionMNIST**: `datasets.FashionMNIST(...)`
- **ImageNet**: `datasets.ImageNet(...)`
- **VOC**: `datasets.VOCDetection(...)` for detection tasks

---

## **5. Data Preprocessing for Text**

### **Tokenization and Embedding**
For text data, tokenization is often the first step, followed by padding and embedding.

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build Vocabulary
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
```

### **Text to Tensor**
Convert text to tensor indices for embedding lookup.

```python
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]
```

---

## **6. Advanced Preprocessing Tips**

### **Handling Imbalanced Classes**
To handle imbalanced classes, use a `WeightedRandomSampler`:

```python
from torch.utils.data import WeightedRandomSampler

# Compute weights for each class
class_counts = [100, 200, 300]  # Example class counts
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights[labels]   # Label is a tensor of dataset labels

# Create sampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))

# DataLoader with sampler
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### **Image Augmentation Libraries**
For extensive image augmentation, consider using `albumentations` or `imgaug`.

```python
from albumentations import Compose, RandomCrop, HorizontalFlip
from albumentations.pytorch import ToTensorV2

transform = Compose([
    RandomCrop(width=256, height=256),
    HorizontalFlip(p=0.5),
    ToTensorV2()
])

# Apply transform within Dataset
class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label
```

---

## **7. DataLoader Tips for Efficiency**

### **Prefetching with `pin_memory=True`**
Set `pin_memory=True` in the `DataLoader` to improve GPU performance by reducing data transfer time.

```python
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
```

### **Use `Persistent Workers` in DataLoader**
Setting `persistent_workers=True` retains worker processes after each epoch, saving time in setting up new workers.

```python
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)
```

### **Load in Parallel**
When working with large datasets, set `num_workers` to at least the number of CPU cores to load data in parallel.

---

## **8. Example: Complete Image Dataset Setup**

```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='data/', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# Example of iterating through the DataLoader
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    # Process batch
```
