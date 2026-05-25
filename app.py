import streamlit as st
import pandas as pd

from calculations.preprocessing import get_inputs
from calculations.forward_propagation import forward_pass
from calculations.error_calculation import calculate_mse
from calculations.backpropagation import backpropagation
from calculations.weight_updates import update_weights


st.title("Titanic Neural Network Manual Calculation")


# Dataset display only
df = pd.read_csv("Titanic-Dataset.csv")

st.header("Titanic Dataset")
st.dataframe(df.head())


# Fixed values from question
x1,x2,x3=get_inputs()

target=1


st.header("Normalized Inputs")

st.write("Pclass =",x1)
st.write("Age =",x2)
st.write("Fare =",x3)


results=forward_pass(
    x1,
    x2,
    x3
)


st.header("Task 1")

st.write(
    "h1 Net Input =",
    round(results["h1_net"],6)
)

st.write(
    "h2 Net Input =",
    round(results["h2_net"],6)
)


st.header("Task 2")

st.write(
    "h1 Output =",
    round(results["h1_out"],6)
)

st.write(
    "h2 Output =",
    round(results["h2_out"],6)
)


st.header("Task 3")

st.write(
    "Output Net Input =",
    round(results["o_net"],6)
)

st.write(
    "Predicted Output =",
    round(results["output"],6)
)


st.header("Task 4")

mse=calculate_mse(
    target,
    results["output"]
)

st.write(
    "MSE =",
    round(mse,6)
)


st.header("Task 5")

gradients=backpropagation(
    target,
    results
)

st.write(
    "Output Gradient =",
    round(
        gradients["output_delta"],
        6
    )
)

st.write(
    "h1 Gradient =",
    round(
        gradients["h1_delta"],
        6
    )
)

st.write(
    "h2 Gradient =",
    round(
        gradients["h2_delta"],
        6
    )
)


st.header("Task 6")

updated=update_weights(
    x1,
    x2,
    x3,
    results,
    gradients
)

for key,value in updated.items():

    st.write(
        key,
        "=",
        round(value,6)
    )