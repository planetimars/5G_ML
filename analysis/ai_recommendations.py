"""AI-based optimization recommendations using Groq with rule-based fallback."""

import json
import os

import pandas as pd
import requests
import streamlit as st

from analysis.optimization_strategies import generate_optimization_plan


GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-70b-versatile"


def _get_groq_api_key():
    key = None
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GROQ_API_KEY")


def has_groq_api_key():
    """Return whether Groq API key is configured in secrets or environment."""
    return bool(_get_groq_api_key())


def _build_prompt(input_df, results_df):
    summary = {
        "input_averages": input_df.mean(numeric_only=True).to_dict(),
        "prediction_averages": {
            "predicted_latency_ms": float(results_df["predicted_latency_ms"].mean()),
            "predicted_throughput_mbps": float(results_df["predicted_throughput_mbps"].mean()),
        },
        "most_common_qos": str(results_df["qos_classification"].mode()[0]),
    }

    return (
        "You are a 5G network optimization assistant. "
        "Provide precise, operator-actionable recommendations as JSON array only. "
        "Each array item must contain keys: Priority, Issue, Current, Target, PreciseAction, EstimatedImpact. "
        "Keep 4-8 items max and avoid vague advice. Use measurable targets.\n\n"
        f"Network summary:\n{json.dumps(summary, indent=2)}"
    )


def _call_groq(prompt, api_key):
    payload = {
        "model": DEFAULT_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=25)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return content


def _parse_recommendations_json(content):
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Groq response is not a list")

    required = ["Priority", "Issue", "Current", "Target", "PreciseAction", "EstimatedImpact"]
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rows.append({k: str(item.get(k, "")) for k in required})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid recommendation rows parsed")
    return df


def generate_ai_optimization_plan(input_df, results_df):
    """Try Groq recommendations first; fallback to deterministic rules."""
    api_key = _get_groq_api_key()
    if not api_key:
        return generate_optimization_plan(input_df, results_df), "fallback"

    try:
        prompt = _build_prompt(input_df, results_df)
        content = _call_groq(prompt, api_key)
        rec_df = _parse_recommendations_json(content)
        return rec_df, "ai"
    except Exception:
        return generate_optimization_plan(input_df, results_df), "fallback"
