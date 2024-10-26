# %%
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.values
y= y.astype(int).values

# %%
print(X.shape)
print(y.shape)

# %%
# Now we should normalize the pixel values
X = ((X / 255.) - .5) * 2

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()
for i in range(25):
    img = X[y == 7][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# %%

from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(X,y, test_size=10000, random_state=123, stratify=y)

X_train, X_Valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)
