import numpy as np

# Initial parameters

w_x1_h1 = 0.11
w_x2_h1 = 0.14
w_x3_h1 = 0.17

w_x1_h2 = 0.21
w_x2_h2 = 0.24
w_x3_h2 = 0.27

b_h1 = 0.1
b_h2 = 0.1

w_h1_o1 = 0.31
w_h2_o1 = 0.34

b_o = 0.1


def sigmoid(x):
    return 1/(1+np.exp(-x))


def predict(x1, x2, x3):

    # Hidden layer

    net_h1 = (
        (x1*w_x1_h1)
        +(x2*w_x2_h1)
        +(x3*w_x3_h1)
        +b_h1
    )

    net_h2 = (
        (x1*w_x1_h2)
        +(x2*w_x2_h2)
        +(x3*w_x3_h2)
        +b_h2
    )

    out_h1 = sigmoid(net_h1)
    out_h2 = sigmoid(net_h2)

    # Output layer

    net_o1 = (
        (out_h1*w_h1_o1)
        +(out_h2*w_h2_o1)
        +b_o
    )

    predicted_output = sigmoid(net_o1)

    return {
        "h1": out_h1,
        "h2": out_h2,
        "prediction": predicted_output
    }