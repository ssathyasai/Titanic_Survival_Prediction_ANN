import streamlit as st
from model import predict

st.set_page_config(
    page_title="Neural Network Demo",
    layout="centered"
)

st.title("Neural Network Forward Propagation")

st.write(
    "Manual Neural Network implementation"
)

x1 = st.slider(
    "Input x1",
    0.0,
    1.0,
    0.2
)

x2 = st.slider(
    "Input x2",
    0.0,
    1.0,
    0.24
)

x3 = st.slider(
    "Input x3",
    0.0,
    1.0,
    0.80
)

if st.button("Predict"):

    result = predict(
        x1,
        x2,
        x3
    )

    st.subheader(
        "Results"
    )

    st.write(
        f"Hidden Neuron h1: {result['h1']:.4f}"
    )

    st.write(
        f"Hidden Neuron h2: {result['h2']:.4f}"
    )

    st.write(
        f"Predicted Output: {result['prediction']:.4f}"
    )

    if result['prediction'] > 0.5:
        st.success(
            "Prediction: Class = 1"
        )
    else:
        st.error(
            "Prediction: Class = 0"
        )