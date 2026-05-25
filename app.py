import streamlit as st
from calculations.preprocessing import get_inputs
from calculations.forward_propagation import forward_pass
from calculations.error_calculation import calculate_mse
from calculations.backpropagation import backpropagation
from calculations.weight_updates import update_weights

st.title("Titanic Neural Network Assignment")

x1, x2, x3 = get_inputs()

st.header("Normalized Inputs")

st.write("Pclass =", x1)
st.write("Age =", x2)
st.write("Fare =", x3)

target = 1

st.header("Task 1 & 2")

results = forward_pass(x1, x2, x3)

st.write("Net Input h1 =", results["h1_net"])
st.write("Output h1 =", results["h1_out"])

st.write("Net Input h2 =", results["h2_net"])
st.write("Output h2 =", results["h2_out"])

st.header("Task 3")

st.write("Output Net Input =", results["o_net"])
st.write("Final Prediction =", results["output"])

st.header("Task 4")

mse = calculate_mse(target, results["output"])

st.write("MSE =", mse)

st.header("Task 5")

gradients = backpropagation(target, results)

st.write("Output Gradient =", gradients["output_delta"])
st.write("Hidden h1 Gradient =", gradients["h1_delta"])
st.write("Hidden h2 Gradient =", gradients["h2_delta"])

st.header("Task 6")

updated_weights = update_weights(
    x1,
    x2,
    x3,
    results,
    gradients
)

for key,value in updated_weights.items():
    st.write(key, "=", value)