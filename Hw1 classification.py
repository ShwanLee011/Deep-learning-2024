
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
# Initialize parameters
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid function

    return A1, A2

# Backward propagation
def backward_propagation(X, Y, A1, A2, W2):
    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Loss function
def compute_loss(A2, Y):
    m = Y.shape[1]
    logprobs = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
    loss = -np.sum(logprobs) / m
    return loss

# Calculate accuracy
def compute_accuracy(A2, Y):
    predictions = (A2 > 0.5).astype(int)  # Set threshold at 0.5
    accuracy = np.mean(predictions == Y)  # Calculate accuracy
    return accuracy

# Train model
def model(X, Y, n_h, epoch, W1, b1, W2, b2, learning_rate=0.001):
    for i in range(epoch):
        # Forward propagation
        A1, A2 = forward_propagation(X, W1, b1, W2, b2)

        # Compute loss
        loss = compute_loss(A2, Y)

        # Backward propagation
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2)

        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    return W1, b1, W2, b2, loss, A2

# Plot latent feature distribution
def plot_latent_features(A1, Y, title):
    pca = PCA(n_components=2)
    latent_features = pca.fit_transform(A1.T)

    plt.figure()
    plt.scatter(latent_features[Y.flatten() == 0, 0], latent_features[Y.flatten() == 0, 1], color='red', label='Class 1')
    plt.scatter(latent_features[Y.flatten() == 1, 0], latent_features[Y.flatten() == 1, 1], color='blue', label='Class 2')
    plt.title(title)
    plt.legend()
    plt.show()

# Plot loss curve
def plot_loss_curve(losses):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

# Main program
if __name__ == "__main__":
    # Load dataset
    data = np.loadtxt('./2024_ionosphere_data.csv', delimiter=',', dtype=str)

    X = data[:, :-1].astype(float).T  # Features (34 features)
    Y = (data[:, -1] == 'g').astype(int).reshape(1, -1)  # Class labels (g is 1, b is 0)

    # Split into training and test sets
    m = X.shape[1]
    m_train = int(0.8 * m)
    X_train, X_test = X[:, :m_train], X[:, m_train:]
    Y_train, Y_test = Y[:, :m_train], Y[:, m_train:]

    # Set hidden layer sizes
    hidden_layer_sizes = [5, 10, 15]  # Multiple hidden layer sizes
    results = []
    # Train the model, plotting every 50 epochs, and showing loss and accuracy
    for n_h in hidden_layer_sizes:
        print(f"Training model with {n_h} hidden neurons...")

        # Initialize weights and biases
        W1, b1, W2, b2 = initialize_parameters(X_train.shape[0], n_h, 1)

        # Training loop
        losses = []
        for epoch in range(1, 1001):  # Train 1000 times
            # Forward and backward propagation to update parameters
            W1, b1, W2, b2, loss, A2_train = model(X_train, Y_train, n_h=n_h, epoch=1, W1=W1, b1=b1, W2=W2, b2=b2)
            losses.append(loss)  # Record loss value each time

            # Plot latent features distribution and show loss and accuracy when epoch# = 20 and 1000
            if epoch == 20 or epoch == 1000:
                train_accuracy = compute_accuracy(A2_train, Y_train)

                # Forward propagation to get predictions for the test set
                _, A2_test = forward_propagation(X_test, W1, b1, W2, b2)
                test_accuracy = compute_accuracy(A2_test, Y_test)

                print(f"Epoch {epoch}, Loss: {loss}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

                # Use forward propagation to obtain latent features
                A1_train, _ = forward_propagation(X_train, W1, b1, W2, b2)
                results.append([n_h, epoch, loss, 1 - train_accuracy, 1 - test_accuracy])

                # Plot 2D distribution of latent features
                plot_latent_features(A1_train, Y_train, f'Latent Features ({n_h} neurons, {epoch} epochs)')

        # Plot loss curve
        plot_loss_curve(losses)
    
    df_results = pd.DataFrame(results, columns=["Hidden Neurons", "Epoch", "Loss", "Training Error Rate", "Test Error Rate"])
    print(df_results)
