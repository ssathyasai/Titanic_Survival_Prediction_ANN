def update_weights(
        x1,
        x2,
        x3,
        results,
        gradients
):


    lr=0.1


    h1=results["h1_out"]
    h2=results["h2_out"]


    output_delta=gradients["output_delta"]

    h1_delta=gradients["h1_delta"]
    h2_delta=gradients["h2_delta"]


    w7=0.31
    w8=0.34


    # Hidden → Output updates

    new_w7=w7+(lr*output_delta*h1)

    new_w8=w8+(lr*output_delta*h2)


    # Input → Hidden updates

    new_w1=0.11+(lr*h1_delta*x1)

    new_w2=0.14+(lr*h1_delta*x2)

    new_w3=0.17+(lr*h1_delta*x3)


    new_w4=0.21+(lr*h2_delta*x1)

    new_w5=0.24+(lr*h2_delta*x2)

    new_w6=0.27+(lr*h2_delta*x3)


    # Bias updates

    new_b1=0.1+(lr*h1_delta)

    new_b2=0.1+(lr*h2_delta)

    new_bo=0.1+(lr*output_delta)



    return{

        "Updated w1":new_w1,
        "Updated w2":new_w2,
        "Updated w3":new_w3,

        "Updated w4":new_w4,
        "Updated w5":new_w5,
        "Updated w6":new_w6,

        "Updated w7":new_w7,
        "Updated w8":new_w8,

        "Updated b1":new_b1,
        "Updated b2":new_b2,
        "Updated bo":new_bo

    }