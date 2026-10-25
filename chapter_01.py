import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}

# Known parameters used to create the data
WEIGHT = 0.7
BIAS = 0.3

  # Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)  
    

def data(show=True):
    # Create data
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = WEIGHT * X + BIAS

    if show:
        print(f"{'X':>8} {'y':>8}")
        for x_val, y_val in zip(X[:10], y[:10]):
            print(f"{x_val.item():>8.4f} {y_val.item():>8.4f}")
    
    # Create train/test split
    train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    if show:
        print(len(X_train), len(y_train), len(X_test), len(y_test))

    return X_train, y_train, X_test, y_test

  
def plot_predictions(train_data, 
                     train_labels, 
                     test_data, 
                     test_labels, 
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="c", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14});
    plt.show()
      
def simple_regression():
    X_train, y_train, X_test, y_test = data()
    plot_predictions(X_train, y_train, X_test, y_test)
    
    # Set manual seed since nn.Parameter are randomly initialized
    torch.manual_seed(42)

    # Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
    model_0 = LinearRegressionModel()

    # Check the nn.Parameter(s) within the nn.Module subclass we created and named parameters
    print(list(model_0.parameters()))
    print(model_0.state_dict())

    # Make predictions with model
    with torch.inference_mode(): 
        y_preds = model_0(X_test)

    # Check the predictions
    print(f"Number of testing samples: {len(X_test)}") 
    print(f"Number of predictions made: {len(y_preds)}")
    print(f"{'X_test':>8} {'y_test':>8} {'y_preds':>8}")
    for x_val, test_val, pred_val in zip(X_test, y_test, y_preds):
        print(f"{x_val.item():>8.2f} {test_val.item():>8.4f} {pred_val.item():>8.4f}")

    plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)
    
def train():
    
    torch.manual_seed(42)
    
    # Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
    X_train, y_train, X_test, y_test = data(False)
    cur_model = LinearRegressionModel()
    
    # Create the loss function
    loss_fn = nn.L1Loss() # MAE loss is same as L1Loss
    
    # Create the optimizer
    # The optimizer is the only thing that changes the model's parameters (weights and bias) during training.
    # params: references to the model parameters it should update
    # lr (learning rate): step size of each update
    #   too large -> parameters overshoot, loss bounces around or diverges
    #   too small -> training needs many epochs to make progress
    # SGD update rule: param = param - lr * param.grad
    # Other optimizers (e.g. torch.optim.Adam) use the same zero_grad()/step() calls with a smarter update rule
    optimizer = torch.optim.SGD(params=cur_model.parameters(), lr=0.01) 

    # Set the number of epochs (how many times the model will pass over the training data)
    epochs = 100

    # Create empty loss lists to track values
    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    for epoch in range(epochs):
        ### Training

        # Put model in training mode (this is the default state of a model)
        cur_model.train()

        # 1. Forward pass on train data using the forward() method inside (uses the current weights and bias)
        y_pred = cur_model(X_train)
        # print(y_pred)

        # 2. Calculate the loss (how different are our models predictions to the ground truth)
        loss = loss_fn(y_pred, y_train)

        # 3. Zero grad of the optimizer
        # Clears the gradients from the previous epoch; PyTorch adds new gradients to old ones by default
        optimizer.zero_grad()

        # 4. Loss backwards
        # Computes the gradient of the loss for each parameter and stores it in param.grad
        # (the gradient points in the direction that increases the loss)
        # Backpropagation only computes the gradients and changes no parameters. The optimizer makes the change, in optimizer.step().
        loss.backward()

        # 5. Progress the optimizer
        # Updates each parameter one step against its gradient (param = param - lr * param.grad), lowering the loss
        # This is: gradient descent (update parameters)
        optimizer.step()

        ### Testing

        # Put the model in evaluation mode
        cur_model.eval()

        with torch.inference_mode():
            # 1. Forward pass on test data
            test_pred = cur_model(X_test)

            # 2. Caculate loss on test data
            test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

        # Print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
                    
    # Plot the loss curves
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend();
    plt.show()
    
    # Find our model's learned parameters
    print("The model learned the following values for weights and bias:")
    print(cur_model.state_dict())
    print("\nAnd the original values for weights and bias are:")
    print(f"weights: {WEIGHT}, bias: {BIAS}")

    return cur_model,  X_train, y_train, X_test, y_test

def workflow():
    # simple_regression()
    trained_model, X_train, y_train, X_test, y_test = train()
    
    trained_model.eval()
    with torch.inference_mode():
        y_preds = trained_model(X_test)

    plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)
    
workflow()