import numpy as np


def sigmoid_activation_function(z):
    return 1/(1 + np.exp(-z))


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    W2 = np.random.randn(n_y, n_h)
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def forward_prop(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid_activation_function(Z2)

    cache = {
        "A1": A1,
        "A2": A2
    }
    return A2, cache


def calculate_cost(A2, Y):
    cost = -np.sum(np.multiply(Y, np.log(A2)) +
                   np.multiply(1-Y, np.log(1-A2)))/m
    cost = np.squeeze(cost)

    return cost


def backward_prop(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]

    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads


def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    new_parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }
    return new_parameters


def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_of_iters+1):
        a2, cache = forward_prop(X, parameters)
        grads = backward_prop(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)
    return parameters


def predict(X, parameters):
    a2, cache = forward_prop(X, parameters)
    yhat = a2
    yhat = np.squeeze(yhat)
    if(yhat >= 0.5):
        y_predict = 1
    else:
        y_predict = 0

    return y_predict


np.random.seed(2)
input = np.array([[22.3, 17.1, 17.32, 21.90, 23.30, 21.175], [6.50, 5.0, 4.73, 6.36, 6.0, 5.72]])
output = np.array([[0, 0, 0, 1, 1, 1]])
m = input.shape[1]
number_input_neurons = 2
number_hidden_neurons = 4
number_output_neurons = 1
number_of_iterations = 700
learning_rate = 0.3

trained_parameters = model(input, output, number_input_neurons, number_hidden_neurons,
                           number_output_neurons, number_of_iterations, learning_rate)


for i in range(1,5):
    X_test = np.array([[input[0][i]], [input[1][i]]])
    y_predict = predict(X_test, trained_parameters)
    print('NN output {:d} for input ({:f}, {:f})'.format(y_predict, X_test[0][0], X_test[1][0]))
