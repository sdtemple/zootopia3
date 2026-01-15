import sys
from datasets import Dataset, Features, Image, ClassLabel
import numpy as np

X_file, y_file, z_file = sys.argv[1:]
X = np.load(X_file)
y = np.loadtxt(y_file, dtype=str)
z = np.loadtxt(z_file, dtype=str)
classes_y = list(np.unique(y))
classes_z = list(np.unique(z))
y = list(y)
z = list(z)
y = [str(_) for _ in y]
z = [str(_) for _ in z]
classes_y = [str(_) for _ in classes_y]
classes_z = [str(_) for _ in classes_z]

# Assuming your data is in numpy arrays: images_np (N, H, W, 3) and labels_np (N,)
def data_generator():
    for img, label, label2 in zip(X, y, z):
        yield {"image": img, "color": label, 'shape': label2}

# Define features: explicitly set 'image' as an Image type
features = Features({
    "image": Image(), 
    "color": ClassLabel(names=classes_y),
    "shape": ClassLabel(names=classes_z),
    }
)

# Create and push
ds = Dataset.from_generator(data_generator, features=features)
ds.push_to_hub("sdtemple/colored-shapes")
