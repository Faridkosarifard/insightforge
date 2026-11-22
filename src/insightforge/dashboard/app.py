"""Streamlit UI: upload a CSV, run EDA, train a baseline, and predict."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

API = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
TIMEOUT = 60

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge")
st.caption("Upload a CSV, explore it, train a baseline model, and run predictions.")


def _format_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    detail = payload.get("detail", payload)
    return str(detail)


def list_datasets() -> list[dict[str, Any]]:
    try:
        response = requests.get(f"{API}/datasets", timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API}: {exc}")
        return []
    if not isinstance(payload, list):
        st.error("Unexpected response from /datasets")
        return []
    return payload


def dataset_label(row: dict[str, Any]) -> str:
    target = row.get("target") or "no target"
    return f"{row['id']}: {row['name']} ({target})"


try:
    health = requests.get(f"{API}/health", timeout=5)
    if health.ok and health.json().get("status") == "ok":
        st.sidebar.success(f"API connected  ·  {API}")
    else:
        st.sidebar.warning(f"API at {API} is not healthy")
except requests.RequestException:
    st.sidebar.error(f"Cannot reach the API at {API}")

tab_upload, tab_eda, tab_train = st.tabs(["Upload", "EDA", "Train & Predict"])

with tab_upload:
    st.subheader("Upload a dataset")
    name = st.text_input("Dataset name", "iris")
    target = st.text_input("Target column (optional)", "species")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if st.button("Upload", type="primary") and uploaded is not None:
        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        params: dict[str, str] = {"name": name}
        if target.strip():
            params["target"] = target.strip()
        try:
            response = requests.post(f"{API}/datasets", files=files, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
        else:
            if response.ok:
                st.success("Dataset stored")
                st.json(response.json())
            else:
                st.error(_format_error(response))

with tab_eda:
    st.subheader("Exploratory analysis")
    datasets = list_datasets()
    options = {dataset_label(row): row["id"] for row in datasets}
    choice = st.selectbox("Dataset", list(options.keys()) if options else ["No datasets yet"])
    if options and choice in options and st.button("Run EDA", type="primary"):
        dataset_id = options[choice]
        try:
            response = requests.post(f"{API}/eda/{dataset_id}", timeout=TIMEOUT)
        except requests.RequestException as exc:
            st.error(f"EDA failed: {exc}")
        else:
            if response.ok:
                body = response.json()
                st.json(body.get("report_json", body.get("report", body)))
            else:
                st.error(_format_error(response))

with tab_train:
    st.subheader("Train a baseline and score new rows")
    datasets = list_datasets()
    options = {dataset_label(row): row["id"] for row in datasets}
    choice = st.selectbox(
        "Dataset",
        list(options.keys()) if options else ["No datasets yet"],
        key="train_dataset",
    )

    col_train, col_predict = st.columns(2)
    with col_train:
        if options and choice in options and st.button("Train baseline model", type="primary"):
            dataset_id = options[choice]
            try:
                response = requests.post(
                    f"{API}/train/{dataset_id}",
                    params={"n_estimators": 200},
                    timeout=TIMEOUT,
                )
            except requests.RequestException as exc:
                st.error(f"Training failed: {exc}")
            else:
                if response.ok:
                    st.success("Model trained")
                    st.json(response.json())
                else:
                    st.error(_format_error(response))

    with col_predict:
        st.write("Predict")
        example = '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]'
        raw = st.text_area("JSON rows (array of objects)", value=example, height=160)
        if options and choice in options and st.button("Send predict"):
            dataset_id = options[choice]
            try:
                rows = json.loads(raw)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
            else:
                if not isinstance(rows, list):
                    st.error("JSON must be an array of row objects")
                else:
                    try:
                        response = requests.post(
                            f"{API}/predict/{dataset_id}",
                            json=rows,
                            timeout=TIMEOUT,
                        )
                    except requests.RequestException as exc:
                        st.error(f"Predict failed: {exc}")
                    else:
                        if response.ok:
                            st.json(response.json())
                        else:
                            st.error(_format_error(response))
