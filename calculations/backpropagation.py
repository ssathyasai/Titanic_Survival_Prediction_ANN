def backpropagation(target,results):

    output=results["output"]

    h1=results["h1_out"]
    h2=results["h2_out"]

    w7=results["w7"]
    w8=results["w8"]

    output_delta=(target-output)*output*(1-output)

    h1_delta=h1*(1-h1)*w7*output_delta

    h2_delta=h2*(1-h2)*w8*output_delta

    return{

        "output_delta":output_delta,

        "h1_delta":h1_delta,

        "h2_delta":h2_delta
    }