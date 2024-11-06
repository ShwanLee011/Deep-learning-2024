import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
f = pd.read_csv('./2024_energy_efficiency_data.csv')

def preprocess_data(filepath):
    # 1. Get the categorical features for one-hot encoding
    orientation_col = f['Orientation'].values
    glazing_area_distribution_col = f['Glazing Area Distribution'].values

    # 2. Get unique categories
    orientations = sorted(set(orientation_col))
    glazing_area_distributions = sorted(set(glazing_area_distribution_col))

    # 3. Create a new DataFrame to store the results of one-hot encoding
    orientation_dummies = np.zeros((len(f), len(orientations)))
    glazing_area_dummies = np.zeros((len(f), len(glazing_area_distributions)))

    # 4. Perform one-hot encoding on the 'Orientation' column
    for i in range(len(f)):
        current_orientation = orientation_col[i]
        orient_index = orientations.index(current_orientation)
        orientation_dummies[i, orient_index] = 1

    # 5. Perform one-hot encoding on the 'Glazing Area Distribution' column
    for i in range(len(f)):
        current_distribution = glazing_area_distribution_col[i]
        glazing_index = glazing_area_distributions.index(current_distribution)
        glazing_area_dummies[i, glazing_index] = 1

    # 6. Convert the one-hot encoded results to DataFrame
    orientation_dummies_df = pd.DataFrame(orientation_dummies, columns=orientations)
    glazing_area_dummies_df = pd.DataFrame(glazing_area_dummies, columns=glazing_area_distributions)

    # 7. Merge the one-hot encoded DataFrame with the original data
    f_encoded = pd.concat([f, orientation_dummies_df, glazing_area_dummies_df], axis=1)

    # 8. Drop the original categorical columns
    f_encoded = f_encoded.drop(['Orientation', 'Glazing Area Distribution'], axis=1)

    # Shuffle the original order
    shuffled = f_encoded.sample(frac=1, random_state=0).reset_index(drop=True)
    trainsize = int(0.75 * len(shuffled))
    trainset = shuffled[:trainsize]
    testset = shuffled[trainsize:]

    # Separate features (X) and target (y) sets
    X_train = trainset.drop(columns=['Heating Load']).values
    y_train = trainset['Heating Load'].values
    X_test = testset.drop(columns=['Heating Load']).values
    y_test = testset['Heating Load'].values
    return X_train, y_train, X_test, y_test

def activation_func(x, actfunc, diff=False):
    if actfunc == 'sigmoid':
        if diff:
            return np.exp(-x) / ((1 + np.exp(-x)) ** 2)
        else:
            return 1 / (1 + np.exp(-x))
    if actfunc == 'relu':
        if diff:
            return np.where(x > 0, 1, 0)
        else:
            return np.maximum(0, x)
    if actfunc == 'tanh':
        if diff:
            return 1 - np.tanh(x) ** 2
        else:
            return np.tanh(x)

class NN:
    def __init__(self, layer_dim, learning_rate, batchsize, epochs, activation='tanh'):
        self.layers = 3
        self.layer_dim = layer_dim
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        self.epochs = epochs
        self.activation = activation  # Store the activation function choice
        self.weights, self.biases = self.initialize_parameters()

    def initialize_parameters(self):
        weights = []
        biases = []
        for i in range(self.layers):
            weights.append(np.random.randn(self.layer_dim[i][0], self.layer_dim[i][1]) * 0.01)
            biases.append(np.random.rand(self.layer_dim[i][1]))
        return weights, biases

    def rms_sum(self, Y, h):
        return np.sum(np.square(Y - h))

    def rmse(self, Y, h):
        return math.sqrt(np.mean(np.square(Y - h)))

    def forward(self, X):
        b_act = []
        a_act = [X]
        for i in range(self.layers):
            btemp = np.dot(a_act[i], self.weights[i]) + self.biases[i]
            b_act.append(btemp)
            if i < self.layers - 1:
                btemp = activation_func(btemp, self.activation)
            a_act.append(btemp)
        return b_act, a_act

    def back_prop(self, b_act, a_act, y):
        w_grad = []
        delta = a_act[-1] - y.reshape(-1, 1)
        for i in range(self.layers):
            w_grad.insert(0, np.dot(a_act[-2-i].T, delta))
            delta = np.dot(delta, self.weights[-1-i].T)
        b_grad = [np.mean(2 * (a_act[-1] - y), axis=0)]
        b_grad.insert(0, np.mean(activation_func(a_act[2], actfunc=self.activation, diff=True), axis=0))
        b_grad.insert(0, np.mean(activation_func(a_act[1], actfunc=self.activation, diff=True), axis=0))
        return w_grad, b_grad

    def update(self, w_grad, b_grad, batchsize):
        for i in range(self.layers):
            self.weights[i] -= self.learning_rate * w_grad[i] / batchsize
            self.biases[i] -= self.learning_rate * np.mean(b_grad[i], axis=0) / batchsize

    def train(self, X_train, y_train, X_test, y_test):
        input_no = X_train.shape[0]
        batch_no = input_no // self.batchsize
        last_batch = input_no % self.batchsize
        last_batch_on = last_batch > 0

        trainloss = []
        testloss = []  # 新增testloss列表來記錄每個epoch的測試損失
        train_perf = []
        test_perf = []

        for epoch in range(self.epochs):
            epoch_train_loss = 0
            y_train_pred = np.zeros_like(y_train)

            for k in range(batch_no):
                X_batch = X_train[k * self.batchsize: (k + 1) * self.batchsize]
                y_batch = y_train[k * self.batchsize: (k + 1) * self.batchsize]

                b_act, a_act = self.forward(X_batch)
                train_loss = self.rms_sum(y_batch, a_act[-1])
                epoch_train_loss += train_loss

                y_train_pred[k * self.batchsize: (k + 1) * self.batchsize] = a_act[-1].reshape(-1)

                w_grad, b_grad = self.back_prop(b_act, a_act, y_batch)
                self.update(w_grad, b_grad, self.batchsize)

            if last_batch_on:
                X_last_batch = X_train[batch_no * self.batchsize:]
                y_last_batch = y_train[batch_no * self.batchsize:]

                b_act, a_act = self.forward(X_last_batch)
                train_loss = self.rms_sum(y_last_batch, a_act[-1])
                epoch_train_loss += train_loss

                y_train_pred[batch_no * self.batchsize:] = a_act[-1].reshape(-1)

                w_grad, b_grad = self.back_prop(b_act, a_act, y_last_batch)
                self.update(w_grad, b_grad, last_batch)

            trainloss.append(epoch_train_loss / (batch_no + (1 if last_batch_on else 0)))

            # 每個epoch都計算測試損失
            b_act_t, a_act_t = self.forward(X_test)
            test_loss = self.rms_sum(y_test, a_act_t[-1])

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs} - Training Loss: {trainloss[-1]}")

        # 在訓練結束後計算 RMSE 並顯示
        train_rmse = self.rmse(y_train, y_train_pred)
        test_rmse = self.rmse(y_test, a_act_t[-1].reshape(-1))

        print(f"Final Training Loss: {trainloss[-1]}")
        print(f"Train RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")

        return trainloss, train_rmse, test_rmse

    def predict(self, X):
        _, a_act = self.forward(X)
        return a_act[-1]

def plot_results(epochs, Lossing_train,  train_performance, test_performance, result1, result2, Y_train, Y_test):
    epoch_list = [i for i in range(epochs)]
    plt.figure(1)
    plt.plot(epoch_list, Lossing_train, label="Training Loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')

    length = len(result1)
    train_data = [i for i in range(length)]
    plt.figure(2)
    plt.plot(train_data, Y_train, label="Actual", color='blue')
    plt.plot(train_data, result1, label="Predicted", color='red')
    plt.legend()
    plt.xlabel('#th case')
    plt.ylabel('Heating load')
    plt.title('Prediction for Training Data')

    length2 = len(result2)
    test_data = [i for i in range(length2)]
    plt.figure(3)
    plt.plot(test_data, Y_test, label="Actual", color='blue')
    plt.plot(test_data, result2, label="Predicted", color='red')
    plt.legend()
    plt.xlabel('#th case')
    plt.ylabel('Heating load')
    plt.title('Prediction for Testing Data')
    plt.show()

# Define hyperparameters
def main():
    X_train, y_train, X_test, y_test = preprocess_data(f)
    layer_dim = [[X_train.shape[1], 10], [10, 16], [16, 1]]
    learning_rate = 0.001
    epochs = 1500

    # Create a neural network object (Choose tanh or relu)
    nn = NN(layer_dim, learning_rate, batchsize=32, epochs=epochs, activation='sigmoid')  # You can change to 'relu'

    # Train the neural network
    trainloss, train_perf, test_perf = nn.train(X_train, y_train, X_test, y_test)

    # Test predictions
    predictions = nn.predict(X_test)

    # Plot results
    plot_results(epochs, trainloss, train_perf, test_perf, nn.predict(X_train), predictions, y_train, y_test)
    avg_train_rmse = np.mean(train_perf)
    avg_test_rmse = np.mean(test_perf)

    print(f"Average Training RMSE: {avg_train_rmse}")
    print(f"Average Testing RMSE: {avg_test_rmse}")


if __name__ == '__main__':
    main()
