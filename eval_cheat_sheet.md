# **Model Evaluation Cheat Sheet with PyTorch Code, Use Cases, and Interpretation**

---

## **1. Classification Metrics**

**Use these metrics for:** Binary and multi-class classification problems (e.g., spam detection, image classification).

### **Accuracy**
- **When to Use**: For balanced datasets where classes are equally important. Best when false positives and false negatives are equally costly.
- **Interpretation**: Higher values indicate a better model, but accuracy may be misleading for imbalanced datasets.

```python
import torch

def accuracy(y_true, y_pred):
    correct = (y_pred == y_true).float()
    acc = correct.sum() / len(correct)
    return acc.item()
```

### **Precision, Recall, and F1 Score**
- **Precision**: Use when false positives are costly (e.g., spam filtering where marking a legitimate email as spam is costly).
- **Recall**: Use when false negatives are costly (e.g., medical diagnosis where missing a positive case is critical).
- **F1 Score**: Use for imbalanced datasets to balance precision and recall.
- **Interpretation**: Higher precision means fewer false positives; higher recall means fewer false negatives; a high F1 indicates a good balance between them.

```python
def precision_recall_f1(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum().float()
    FP = ((y_pred == 1) & (y_true == 0)).sum().float()
    FN = ((y_pred == 0) & (y_true == 1)).sum().float()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1.item()
```

### **AUC-ROC**
- **When to Use**: For binary classification with probabilistic outputs, especially for imbalanced datasets to measure class separation.
- **Interpretation**: Values closer to 1 indicate a model better at distinguishing between classes.

```python
from sklearn.metrics import roc_auc_score

def auc_roc(y_true, y_pred_proba):
    return roc_auc_score(y_true.cpu().numpy(), y_pred_proba.cpu().numpy())
```

---

## **2. Regression Metrics**

**Use these metrics for:** Continuous-valued predictions (e.g., price prediction, demand forecasting).

### **Mean Absolute Error (MAE)**
- **When to Use**: When all errors are equally important; less sensitive to outliers.
- **Interpretation**: Lower MAE means predictions are closer to the actual values on average.

```python
def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()
```

### **Mean Squared Error (MSE)**
- **When to Use**: When larger errors should be penalized more heavily.
- **Interpretation**: Lower MSE indicates fewer large errors, but it is sensitive to outliers.

```python
def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2).item()
```

### **Root Mean Squared Error (RMSE)**
- **When to Use**: Same as MSE but provides interpretable error units.
- **Interpretation**: Lower RMSE means better performance; it is more sensitive to large errors than MAE.

```python
def root_mean_squared_error(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
```

### **R-Squared (R²)**
- **When to Use**: To understand the proportion of variance explained by the model. Suitable for linear regression.
- **Interpretation**: Values close to 1 indicate a strong model fit, while values close to 0 indicate a poor fit.

```python
def r_squared(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / (ss_total + 1e-8)
    return r2.item()
```

---

## **3. Clustering Metrics**

**Use these metrics for:** Unsupervised tasks, such as customer segmentation or document grouping.

### **Silhouette Score**
- **When to Use**: To assess how well-defined clusters are. Useful for k-means or hierarchical clustering.
- **Interpretation**: Values closer to 1 indicate better-defined clusters, while values close to 0 suggest overlapping clusters.

```python
from sklearn.metrics import silhouette_score

def silhouette(X, labels):
    return silhouette_score(X, labels)
```

### **Adjusted Rand Index (ARI)**
- **When to Use**: For comparing clustering results to a ground truth. Ideal when labeled data is available.
- **Interpretation**: Values closer to 1 show better alignment with true labels; values near 0 indicate poor alignment.

```python
from sklearn.metrics import adjusted_rand_score

def adjusted_rand_index(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)
```

---

## **4. Time Series Forecasting Metrics**

**Use these metrics for:** Predicting future values in sequential data (e.g., sales, temperature).

### **Mean Absolute Percentage Error (MAPE)**
- **When to Use**: When you need interpretability in percentage terms.
- **Interpretation**: Lower MAPE means better performance; however, MAPE can be skewed when actual values are very low.

```python
def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))).item() * 100
```

### **MSE & RMSE**
- **When to Use**: MSE penalizes large errors more, making it useful for general time series forecasting.
- **Interpretation**: Lower MSE or RMSE indicates more accurate predictions, with RMSE offering units that are easier to interpret.

Refer to the **Regression Metrics** section above for code.

---

## **5. Ranking Metrics**

**Use these metrics for:** Ranking and recommendation tasks (e.g., search engines, recommender systems).

### **Mean Average Precision (MAP)**
- **When to Use**: For evaluating ranked lists where relevance is crucial. Often used in recommendation systems.
- **Interpretation**: Values closer to 1 indicate better ranking performance and more relevant items at the top.

```python
from sklearn.metrics import average_precision_score

def mean_average_precision(y_true, y_pred_scores):
    return average_precision_score(y_true.cpu().numpy(), y_pred_scores.cpu().numpy())
```

### **Normalized Discounted Cumulative Gain (NDCG)**
- **When to Use**: For ranking problems where top positions are more important. Ideal for highly skewed relevance data.
- **Interpretation**: Higher values, especially at the top `k` positions, indicate better relevance of top-ranked items.

```python
from sklearn.metrics import ndcg_score

def normalized_discounted_cumulative_gain(y_true, y_pred_scores, k=10):
    return ndcg_score([y_true], [y_pred_scores], k=k)
```
