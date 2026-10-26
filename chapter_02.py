# sklearn must be imported before torch: torch loads the older system libstdc++,
# which lacks the CXXABI_1.3.15 that scipy (used by sklearn) needs
from sklearn.datasets import make_circles
import torch
from torch import nn  # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 feature (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

def data(show=True):

    # Make 1000 samples
    n_samples = 1000

    # Create circles
    X, y = make_circles(
        n_samples, noise=0.03, random_state=42  # a little bit of noise to the dots
    )  # keep random state so we get the same values
    
    if show:
        print(f"First 5 X features:\n{X[:5]}")
        print(f"\nFirst 5 y labels:\n{y[:5]}")

    circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})

    if show:
        print(circles.head(10))
        # Check different labels
        print(circles.label.value_counts())
        # Visualize with a plot
        plt.scatter(x=X[:, 0], 
                    y=X[:, 1], 
                    c=y, 
                    cmap=plt.cm.RdYlBu);
        plt.show()
        
    # Check the shapes of our features and labels
    # Two features for one output: vector to scalar
    print("X-shape", X.shape, "y-shape", y.shape)
    
    # Convert to tensors
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

    print(len(X_train), len(X_test), len(y_train), len(y_test))
    return X_train, X_test, y_train, y_test
    
def make_model():
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cur_model = CircleModelV0().to(device)
    print(cur_model)
    
    
def workflow():       
    X_train, X_test, y_train, y_test = data(False)
    make_model()
    
workflow()