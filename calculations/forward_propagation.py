import math


def sigmoid(x):

    return 1/(1+math.exp(-x))


def forward_pass(x1,x2,x3):


    # Input → Hidden weights

    w1=0.11
    w2=0.14
    w3=0.17

    w4=0.21
    w5=0.24
    w6=0.27


    # Hidden Bias

    b1=0.1
    b2=0.1


    # Hidden → Output weights

    w7=0.31
    w8=0.34


    # Output bias

    bo=0.1


    # Hidden layer net input

    h1_net=(x1*w1)+(x2*w2)+(x3*w3)+b1

    h2_net=(x1*w4)+(x2*w5)+(x3*w6)+b2


    # Hidden output

    h1_out=sigmoid(h1_net)

    h2_out=sigmoid(h2_net)


    # Output layer

    o_net=(h1_out*w7)+(h2_out*w8)+bo

    output=sigmoid(o_net)


    return{

        "h1_net":h1_net,
        "h2_net":h2_net,

        "h1_out":h1_out,
        "h2_out":h2_out,

        "o_net":o_net,
        "output":output,

        "w7":w7,
        "w8":w8
    }