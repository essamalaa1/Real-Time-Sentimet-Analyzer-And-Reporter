import streamlit as st
import json
import os
from config import STATE_FILE, MODEL_OPTIONS

def load_app_state_as_dict():
    default_state = {
        # Batch Reporter settings
        "last_processed_row_index": 0,
        "sheet_id": "",
        "batch_size": 3,
        "selected_cols": [],
        "selected_model_label": list(MODEL_OPTIONS.keys())[0] if MODEL_OPTIONS else "N/A",

        # Sentiment Analyzer settings
        "sentiment_sheet_id": "",
        "sentiment_selected_cols": [],
        "sentiment_last_processed_row_index": 0,
    }
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            merged_state = default_state.copy()
            merged_state.update(state)
            if 'selected_model_label' not in merged_state and MODEL_OPTIONS:
                 merged_state['selected_model_label'] = list(MODEL_OPTIONS.keys())[0]
            return merged_state
        else:
            st.info(f"No state file found at '{STATE_FILE}'. Starting with a fresh state.")
            return default_state
    except json.JSONDecodeError:
        st.warning(f"Warning: State file '{STATE_FILE}' is corrupted. Starting with a fresh state.")
        return default_state
    except Exception as e:
        st.error(f"Error loading state file: {e}. Starting with a fresh state.")
        return default_state

def save_app_state_from_dict(state_dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state_dict, f, indent=4)
    except Exception as e:
        st.error(f"Error saving state to '{STATE_FILE}': {e}")
