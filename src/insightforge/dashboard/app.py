# Standard library
import io
import json
import os

# Third-party (others)
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge Dashboard")

tab1, tab2, tab3 = st.tabs(["Upload", "EDA", "Train & Predict"])

with tab1:
    st.subheader("Upload dataset")
    name = st.text_input("Dataset name", "iris")
    target = st.text_input("Target column (optional)", "")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Upload") and file:
        files = {"file": (file.name, file.getvalue(), "text/csv")}
        params = {"name": name}
        if target:
            params["target"] = target
        r = requests.post(f"{API}/datasets", files=files, params=params)
        st.write(r.status_code)
        st.json(r.json() if r.ok else r.text)

with tab2:
    st.subheader("Run EDA")
    r = requests.get(f"{API}/datasets")
    ds = r.json() if r.ok else []
    options = {f'{d["id"]}: {d["name"]}': d["id"] for d in ds}
    choice = st.selectbox("Choose dataset", list(options.keys())) if options else None
    if choice and st.button("Run EDA"):
        ds_id = options[choice]
        r = requests.post(f"{API}/eda/{ds_id}")
        if r.ok:
            rep = r.json().get("report", r.json().get("report_json"))
            st.json(rep)
        else:
            st.error(r.text)

with tab3:
    st.subheader("Train & Predict")
    r = requests.get(f"{API}/datasets")
    ds = r.json() if r.ok else []
    options = {f'{d["id"]}: {d["name"]}': d["id"] for d in ds}
    choice = (
        st.selectbox("Dataset", list(options.keys()), key="train") if options else None
    )

    col1, col2 = st.columns(2)
    with col1:
        if choice and st.button("Train baseline model"):
            ds_id = options[choice]
            r = requests.post(f"{API}/train/{ds_id}")
            st.json(r.json() if r.ok else r.text)

    with col2:
        st.write("Predict")
        example = '[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]'
        raw = st.text_area("JSON rows", value=example, height=150)
        if choice and st.button("Send predict"):
            ds_id = options[choice]
            try:
                rows = pd.read_json(io.StringIO(raw))
                r = requests.post(
                    f"{API}/predict/{ds_id}", json=rows.to_dict(orient="records")
                )
                st.json(r.json() if r.ok else r.text)
            except Exception as e:
                st.error(str(e))
