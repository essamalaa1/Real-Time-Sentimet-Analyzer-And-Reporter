import streamlit as st
import pandas as pd
import requests
import io
import re
from langchain_ollama import OllamaLLM
from config import SYSTEM_PROMPT_TEMPLATE

@st.cache_data(show_spinner=False)
def fetch_dataframe_cached(sheet_id):
    try:
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        resp = requests.get(csv_url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching data: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing data from Google Sheet: {e}")
        return None

def fetch_dataframe_fresh(sheet_id):
    try:
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        resp = requests.get(csv_url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()
        return df
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching fresh data: {e}") # Use print for background task feedback
        return None
    except Exception as e:
        print(f"Error processing fresh data: {e}") # Use print
        return None

@st.cache_resource
def get_llm(model_name):
    return OllamaLLM(model=model_name, temperature=0.3)

def format_reviews(batch_df, selected_cols_param):
    lines = []
    for _, row in batch_df.iterrows():
        parts = [str(row[c]).strip() for c in selected_cols_param if c in row.index and pd.notnull(row[c])]
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)

def report_batch(start_idx, batch_df, selected_cols_param, llm_param):
    batch_range = f"{start_idx + 1}-{start_idx + len(batch_df)}"
    reviews_text = format_reviews(batch_df, selected_cols_param)
    if not reviews_text:
        return f"⚠️ No valid reviews found in rows {batch_range}. Skipping..."

    current_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(batch_range=batch_range)
    messages = [
        ("system", current_system_prompt),
        ("user", f"Here are the reviews (one per line):\n{reviews_text}")
    ]
    try:
        report = llm_param.invoke(messages)
        expected_title = f"### Report for Batch {batch_range}"
        if not report.strip().startswith(expected_title):
            title_match_no_hash = re.search(f"Report for Batch {re.escape(batch_range)}", report, re.IGNORECASE)
            if title_match_no_hash:
                report = re.sub(f"Report for Batch {re.escape(batch_range)}", expected_title, report, count=1, flags=re.IGNORECASE)
            else:
                report = f"{expected_title}\n\n{report}"

        report = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL) # Remove potential <think> tags
        return report
    except Exception as e:
        st.error(f"Error during LLM invocation for batch {batch_range}: {e}")
        return f"❌ Error generating report for batch {batch_range}: LLM failed. Details: {str(e)[:200]}..."