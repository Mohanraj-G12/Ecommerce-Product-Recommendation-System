# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from backend import get_model_results

st.set_page_config(page_title="Ecommerce Product Recommendation", layout="wide")

st.title("Ecommerce Model Prediction")

# Show dataset info
st.subheader("Dataset Overview")
df = pd.read_csv("Ecommerce Dataset.csv")
st.write(df.head())

# Fetch model results
results = get_model_results()

# Show Accuracy Comparison
st.subheader("Model Accuracy Comparison")
accuracies = {model: res["accuracy"] for model, res in results.items()}

fig, ax = plt.subplots()
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
st.pyplot(fig)

# Classification Reports
st.subheader("Classification Reports")
cols = st.columns(4)
for idx, (model_name, data) in enumerate(results.items()):
    with cols[idx]:
        st.markdown(f"**{model_name}**")
        st.text(data["report_text"])
