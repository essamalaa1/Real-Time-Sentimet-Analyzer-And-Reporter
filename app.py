import streamlit as st
import pandas as pd
import time
import os
import re

from config import (
    STATE_FILE, REFRESH_INTERVAL, MODEL_OPTIONS
)
from pdf_utils import (
    generate_pdf_from_markdown_bytes,
    extract_batch_range_for_filename
)
from state_manager import (
    load_app_state_as_dict,
    save_app_state_from_dict
)
from core_logic import (
    fetch_dataframe_cached,
    fetch_dataframe_fresh,
    get_llm,
    format_reviews,
    report_batch
)
from sentiment_analyzer import (
    predict_sentiments_for_texts,
    load_sentiment_model_and_artifacts
)

# --- Page Config and Initial Setup ---
st.set_page_config(page_title="Realtime Customer Insights Processor", layout="wide")
st.title("📊 Realtime Customer Insights Processor")
st.caption(f"App state is persisted in: {os.path.abspath(STATE_FILE)}")

# --- Session State Initialization ---
if 'app_state_loaded' not in st.session_state:
    loaded_state_dict = load_app_state_as_dict()
    st.session_state.initial_settings = loaded_state_dict
    st.session_state.run_settings = loaded_state_dict.copy()

    st.session_state.batch_reporter_running = False
    st.session_state.sentiment_analyzer_running = False

    st.session_state.reports_generated_this_session = []
    st.session_state.sentiment_processed_df = pd.DataFrame()
    st.session_state.sentiment_display_trigger = 0

    st.session_state.df_preview_cache_reporter = None
    st.session_state.df_preview_cache_sentiment = None

    st.session_state.preview_cache_reporter_sheet_id = ""
    st.session_state.preview_cache_sentiment_sheet_id = ""

    st.session_state.app_state_loaded = True
    
    # Load sentiment model and artifacts once at startup
    # This will be used by Tab 2 (Sentiment Analyzer) and Tab 3 (Ad-hoc Tester)
    load_sentiment_model_and_artifacts()


# --- Helper function to get DataFrame for UI preview ---
def get_df_for_preview(sheet_id_input, cache_key_df, cache_key_sheet_id):
    df_for_ui = None
    if sheet_id_input:
        if st.session_state[cache_key_df] is None or st.session_state[cache_key_sheet_id] != sheet_id_input:
            with st.spinner("Fetching data for preview..."):
                st.session_state[cache_key_df] = fetch_dataframe_cached(sheet_id_input)
                st.session_state[cache_key_sheet_id] = sheet_id_input
        df_for_ui = st.session_state[cache_key_df]
        if df_for_ui is not None:
            st.success(f"Data preview ready. Total rows: {len(df_for_ui)}")
    return df_for_ui

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📋 Batch Reporter", "🧐 Sentiment Analyzer", "💬 Ad-hoc Sentiment Test"])

# ==============================================================================
# TAB 1: BATCH REPORTER
# ==============================================================================
with tab1:
    st.header("📋 Realtime Customer Review Batch Reporter")

    rep_ui_sheet_id = st.text_input(
        "Enter Google Sheet ID (for Batch Reporter):",
        value=st.session_state.initial_settings['sheet_id'],
        key="reporter_sheet_id_input"
    )
    rep_ui_batch_size = st.number_input(
        "Batch Size (process when this many new reviews are available):",
        min_value=1,
        value=st.session_state.initial_settings['batch_size'],
        key="reporter_batch_size_input"
    )

    rep_df_for_ui = get_df_for_preview(rep_ui_sheet_id, 'df_preview_cache_reporter', 'preview_cache_reporter_sheet_id')

    if rep_df_for_ui is not None:
        st.dataframe(rep_df_for_ui.head(), height=150, key="reporter_df_preview")
        rep_available_cols = rep_df_for_ui.columns.tolist()
        rep_persisted_sel_cols = st.session_state.initial_settings['selected_cols']
        rep_valid_persisted_sel_cols = [col for col in rep_persisted_sel_cols if col in rep_available_cols]
        
        rep_ui_selected_cols = st.multiselect(
            "Select columns for LLM (Batch Reporter):",
            options=rep_available_cols,
            default=rep_valid_persisted_sel_cols if rep_valid_persisted_sel_cols else None,
            key="reporter_selected_cols_input"
        )
    else:
        rep_ui_selected_cols = st.multiselect(
            "Select columns for LLM (Batch Reporter):", options=[], default=[],
            help="Enter a valid Sheet ID to populate columns.",
            key="reporter_selected_cols_input_disabled"
        )

    rep_ui_selected_model_label = st.selectbox(
        "Select LLM Model for Reports:",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(st.session_state.initial_settings['selected_model_label'])
              if st.session_state.initial_settings['selected_model_label'] in MODEL_OPTIONS else 0,
        key="reporter_model_select"
    )
    rep_reprocess_all = st.checkbox("🔄 Reprocess all data from the beginning (Reporter)", value=False, key="reporter_reprocess_all")
    st.caption("If checked, Batch Reporter will start from row 1 for the current Sheet ID.")

    st.markdown("---")
    rep_col1, rep_col2 = st.columns(2)
    with rep_col1:
        rep_start_disabled = st.session_state.batch_reporter_running or not rep_ui_sheet_id or not rep_ui_selected_cols
        if st.button("▶️ Start Batch Reporting", disabled=rep_start_disabled, type="primary", key="reporter_start_button"):

            sheet_id_changed = st.session_state.run_settings['sheet_id'] != rep_ui_sheet_id and st.session_state.run_settings['sheet_id'] != ""
            if sheet_id_changed:
                st.session_state.run_settings['last_processed_row_index'] = 0
                st.toast(f"Reporter Sheet ID changed. Processing will start from the beginning of '{rep_ui_sheet_id}'.", icon="ℹ️")
                st.session_state.reports_generated_this_session = []
            elif rep_reprocess_all:
                st.session_state.run_settings['last_processed_row_index'] = 0
                st.toast(f"Reporter re-processing all data for sheet '{rep_ui_sheet_id}' from the beginning.", icon="🔄")
                st.session_state.reports_generated_this_session = []
            
            st.session_state.batch_reporter_running = True
            st.session_state.run_settings['sheet_id'] = rep_ui_sheet_id
            st.session_state.run_settings['batch_size'] = rep_ui_batch_size
            st.session_state.run_settings['selected_cols'] = rep_ui_selected_cols
            st.session_state.run_settings['selected_model_label'] = rep_ui_selected_model_label
            save_app_state_from_dict(st.session_state.run_settings)
            st.rerun()

    with rep_col2:
        if st.button("⏹️ Stop Batch Reporting", disabled=not st.session_state.batch_reporter_running, key="reporter_stop_button"):
            st.session_state.batch_reporter_running = False
            st.toast("Batch reporting stopped by user.", icon="🛑")
            save_app_state_from_dict(st.session_state.run_settings)
            st.rerun()

    st.markdown("---")
    reporter_status_placeholder = st.empty()
    reporter_reports_container = st.container()

    with reporter_reports_container:
        st.subheader("Generated Reports (Newest First)")
        if not st.session_state.reports_generated_this_session:
            st.caption("No reports generated in this session yet.")
        for idx, report_md_content in enumerate(reversed(st.session_state.reports_generated_this_session)):
            report_title_for_expander = f"Report {len(st.session_state.reports_generated_this_session) - idx}"
            title_match = re.search(r"###\s*(Report for Batch\s*[\w\d\s-]+)", report_md_content, re.IGNORECASE)
            if title_match:
                report_title_for_expander = title_match.group(1).strip()

            with st.expander(report_title_for_expander, expanded=(idx == 0)):
                st.markdown(report_md_content, unsafe_allow_html=True)
                pdf_file_name = f"{extract_batch_range_for_filename(report_md_content)}.pdf"
                try:
                    pdf_bytes = generate_pdf_from_markdown_bytes(report_md_content)
                    if pdf_bytes:
                        st.download_button(
                            label="📄 Download Report as PDF", data=pdf_bytes, file_name=pdf_file_name,
                            mime="application/pdf", key=f"pdf_download_reporter_{len(st.session_state.reports_generated_this_session) - idx}"
                        )
                except Exception as e:
                    st.error(f"Could not prepare PDF for download: {e}")

    # --- Batch Reporter Processing Loop ---
    if st.session_state.batch_reporter_running:
        current_reporter_settings = st.session_state.run_settings
        reporter_status_placeholder.info(
            f"🚀 Batch Reporter active for Sheet ID: '{current_reporter_settings['sheet_id']}'. "
            f"Last processed: row {current_reporter_settings['last_processed_row_index']}. Checking..."
        )
        
        llm_for_processing = None
        try:
            llm_for_processing = get_llm(MODEL_OPTIONS[current_reporter_settings['selected_model_label']])
        except KeyError:
            reporter_status_placeholder.error(f"Invalid model '{current_reporter_settings['selected_model_label']}'. Stopping reporter.")
            st.session_state.batch_reporter_running = False
            save_app_state_from_dict(st.session_state.run_settings) # Save state
            st.rerun(); st.stop()
        except Exception as e:
            reporter_status_placeholder.error(f"Error initializing LLM for reporter: {e}. Stopping.")
            st.session_state.batch_reporter_running = False
            save_app_state_from_dict(st.session_state.run_settings) # Save state
            st.rerun(); st.stop()

        fetched_df_reporter = fetch_dataframe_fresh(current_reporter_settings['sheet_id'])
        should_wait_reporter = True

        if fetched_df_reporter is not None:
            missing_cols = [col for col in current_reporter_settings['selected_cols'] if col not in fetched_df_reporter.columns]
            if missing_cols:
                reporter_status_placeholder.error(
                    f"Reporter Error: Selected column(s) {missing_cols} not found in sheet '{current_reporter_settings['sheet_id']}'. Stopping."
                )
                st.session_state.batch_reporter_running = False
                save_app_state_from_dict(st.session_state.run_settings)
                st.rerun(); st.stop()

            start_index = current_reporter_settings['last_processed_row_index']
            num_total_rows = len(fetched_df_reporter)

            if start_index > num_total_rows:
                reporter_status_placeholder.warning(f"Reporter: Last processed row ({start_index}) > sheet length ({num_total_rows}). Resetting to 0.")
                current_reporter_settings['last_processed_row_index'] = 0
                start_index = 0
                save_app_state_from_dict(current_reporter_settings)

            num_new_rows = num_total_rows - start_index
            if num_new_rows >= current_reporter_settings['batch_size']:
                batch_df = fetched_df_reporter.iloc[start_index : start_index + current_reporter_settings['batch_size']]
                batch_range_disp = f"{start_index + 1}-{start_index + len(batch_df)}"
                reporter_status_placeholder.info(f"Reporter: Processing batch for rows: {batch_range_disp}...")
                
                report_md = report_batch(
                    start_index, batch_df,
                    current_reporter_settings['selected_cols'], llm_for_processing
                )
                st.session_state.reports_generated_this_session.append(report_md)
                current_reporter_settings['last_processed_row_index'] = start_index + len(batch_df)
                save_app_state_from_dict(current_reporter_settings)
                
                reporter_status_placeholder.success(
                    f"Reporter: Batch {batch_range_disp} processed. Last processed: {current_reporter_settings['last_processed_row_index']}."
                )
                should_wait_reporter = False
                time.sleep(0.2)
                st.rerun(); st.stop()
            elif num_new_rows > 0:
                needed = current_reporter_settings['batch_size'] - num_new_rows
                reporter_status_placeholder.info(
                    f"Reporter: {num_new_rows} new review(s). Waiting for {needed} more for batch. Next check in {REFRESH_INTERVAL}s."
                )
            else:
                reporter_status_placeholder.info(
                    f"Reporter: No new reviews beyond row {start_index}. Next check in {REFRESH_INTERVAL}s."
                )
        elif fetched_df_reporter is None and st.session_state.batch_reporter_running:
            reporter_status_placeholder.error(
                f"Reporter: Failed to fetch data for Sheet ID: '{current_reporter_settings['sheet_id']}'. Retry in {REFRESH_INTERVAL}s."
            )
        
        if st.session_state.batch_reporter_running and should_wait_reporter:
            time.sleep(REFRESH_INTERVAL)
            st.rerun()

    elif 'app_state_loaded' in st.session_state and not st.session_state.batch_reporter_running:
        reporter_status_placeholder.info(
            f"Batch Reporter is stopped. Configure and click 'Start'. Last processed for '{st.session_state.run_settings.get('sheet_id', 'N/A')}' was row {st.session_state.run_settings.get('last_processed_row_index', 'N/A')}."
        )


# ==============================================================================
# TAB 2: SENTIMENT ANALYZER
# ==============================================================================
with tab2:
    st.header("🧐 Real-time Row Sentiment Analyzer")

    sa_ui_sheet_id = st.text_input(
        "Enter Google Sheet ID (for Sentiment Analysis):",
        value=st.session_state.initial_settings['sentiment_sheet_id'],
        key="sentiment_sheet_id_input"
    )

    sa_df_for_ui = get_df_for_preview(sa_ui_sheet_id, 'df_preview_cache_sentiment', 'preview_cache_sentiment_sheet_id')

    if sa_df_for_ui is not None:
        st.dataframe(sa_df_for_ui.head(), height=150, key="sentiment_df_preview")
        sa_available_cols = sa_df_for_ui.columns.tolist()
        sa_persisted_sel_cols = st.session_state.initial_settings['sentiment_selected_cols']
        sa_valid_persisted_sel_cols = [col for col in sa_persisted_sel_cols if col in sa_available_cols]
        
        sa_ui_selected_cols = st.multiselect(
            "Select columns for Sentiment Analysis:",
            options=sa_available_cols,
            default=sa_valid_persisted_sel_cols if sa_valid_persisted_sel_cols else None,
            key="sentiment_selected_cols_input"
        )
    else:
        sa_ui_selected_cols = st.multiselect(
            "Select columns for Sentiment Analysis:", options=[], default=[],
            help="Enter a valid Sheet ID to populate columns.",
            key="sentiment_selected_cols_input_disabled"
        )

    sa_reprocess_all = st.checkbox("🔄 Reprocess all data from the beginning (Sentiment)", value=False, key="sentiment_reprocess_all")
    st.caption("If checked, Sentiment Analyzer will process all rows from the current Sheet ID.")

    st.markdown("---")
    sa_col1, sa_col2 = st.columns(2)
    with sa_col1:
        sa_start_disabled = st.session_state.sentiment_analyzer_running or not sa_ui_sheet_id or not sa_ui_selected_cols
        if st.button("▶️ Start Sentiment Analysis", disabled=sa_start_disabled, type="primary", key="sentiment_start_button"):

            sheet_id_changed_sa = st.session_state.run_settings['sentiment_sheet_id'] != sa_ui_sheet_id and st.session_state.run_settings['sentiment_sheet_id'] != ""
            if sheet_id_changed_sa or sa_reprocess_all:
                st.session_state.run_settings['sentiment_last_processed_row_index'] = 0
                st.session_state.sentiment_processed_df = pd.DataFrame()
                st.session_state.sentiment_display_trigger += 1
                if sheet_id_changed_sa:
                    st.toast(f"Sentiment Sheet ID changed. Processing will start from the beginning of '{sa_ui_sheet_id}'.", icon="ℹ️")
                else:
                    st.toast(f"Sentiment re-processing all data for sheet '{sa_ui_sheet_id}' from the beginning.", icon="🔄")
            
            st.session_state.sentiment_analyzer_running = True
            st.session_state.run_settings['sentiment_sheet_id'] = sa_ui_sheet_id
            st.session_state.run_settings['sentiment_selected_cols'] = sa_ui_selected_cols

            save_app_state_from_dict(st.session_state.run_settings)
            st.rerun()

    with sa_col2:
        if st.button("⏹️ Stop Sentiment Analysis", disabled=not st.session_state.sentiment_analyzer_running, key="sentiment_stop_button"):
            st.session_state.sentiment_analyzer_running = False
            st.toast("Sentiment analysis stopped by user.", icon="🛑")
            save_app_state_from_dict(st.session_state.run_settings)
            st.rerun()

    st.markdown("---")
    sentiment_status_placeholder = st.empty()
    sentiment_results_container = st.container()
    
    with sentiment_results_container:
        st.subheader("Sentiment Analysis Results (Live)")
        if st.session_state.sentiment_processed_df.empty:
            st.caption("No sentiment data processed yet or processing reset. Click 'Start'.")
        st.dataframe(st.session_state.sentiment_processed_df, key=f"sentiment_results_df_{st.session_state.sentiment_display_trigger}", height=400)


    # --- Sentiment Analyzer Processing Loop ---
    if st.session_state.sentiment_analyzer_running:
        current_sa_settings = st.session_state.run_settings
        sentiment_status_placeholder.info(
            f"🧐 Sentiment Analyzer active for Sheet ID: '{current_sa_settings['sentiment_sheet_id']}'. "
            f"Last processed: row {current_sa_settings['sentiment_last_processed_row_index']}. Checking..."
        )
        
        _model_check, _, _, _, _ = load_sentiment_model_and_artifacts()
        if _model_check is None:
            sentiment_status_placeholder.error("Sentiment model could not be loaded. Stopping Sentiment Analyzer.")
            st.session_state.sentiment_analyzer_running = False
            save_app_state_from_dict(st.session_state.run_settings)
            st.rerun(); st.stop()


        fetched_df_sa = fetch_dataframe_fresh(current_sa_settings['sentiment_sheet_id'])
        should_wait_sa = True

        if fetched_df_sa is not None:
            missing_cols_sa = [col for col in current_sa_settings['sentiment_selected_cols'] if col not in fetched_df_sa.columns]
            if missing_cols_sa:
                sentiment_status_placeholder.error(
                    f"Sentiment Error: Selected column(s) {missing_cols_sa} not found in sheet '{current_sa_settings['sentiment_sheet_id']}'. Stopping."
                )
                st.session_state.sentiment_analyzer_running = False
                save_app_state_from_dict(st.session_state.run_settings)
                st.rerun(); st.stop()

            start_index_sa = current_sa_settings['sentiment_last_processed_row_index']
            num_total_rows_sa = len(fetched_df_sa)

            if start_index_sa > num_total_rows_sa:
                sentiment_status_placeholder.warning(f"Sentiment: Last processed row ({start_index_sa}) > sheet length ({num_total_rows_sa}). Resetting to 0.")
                current_sa_settings['sentiment_last_processed_row_index'] = 0
                start_index_sa = 0
                st.session_state.sentiment_processed_df = pd.DataFrame()
                st.session_state.sentiment_display_trigger += 1
                save_app_state_from_dict(current_sa_settings)

            new_rows_to_process_df = fetched_df_sa.iloc[start_index_sa:]

            if not new_rows_to_process_df.empty:
                sentiment_status_placeholder.info(f"Sentiment: Processing {len(new_rows_to_process_df)} new/updated row(s) from index {start_index_sa}...")
                
                processed_rows_list = []
                if not st.session_state.sentiment_processed_df.empty:
                    # Ensure we only take rows that are *before* the new processing start index
                    processed_rows_list = st.session_state.sentiment_processed_df.head(start_index_sa).to_dict('records')

                temp_new_rows = []
                for idx, row in new_rows_to_process_df.iterrows():
                    row_dict = row.to_dict()
                    texts_for_sentiment = []
                    
                    for col_name in current_sa_settings['sentiment_selected_cols']:
                        if col_name in row and pd.notna(row[col_name]):
                            text_val = str(row[col_name]).strip()
                            if text_val:
                                texts_for_sentiment.append(text_val)
                    
                    # Get sentiment for all selected texts in the row
                    row_sentiments_predictions = predict_sentiments_for_texts(texts_for_sentiment)
                    
                    # Determine overall sentiment for the row
                    # If any text in the selected columns for the row is Negative, the row is Negative.
                    # If all are Positive (and no Negative), the row is Positive.
                    # Otherwise N/A or Error.
                    overall_row_sentiment = "N/A" # Default
                    if any("Error:" in s for s in row_sentiments_predictions):
                        overall_row_sentiment = "Error"
                    elif "Negative" in row_sentiments_predictions:
                        overall_row_sentiment = "Negative"
                    elif all(s == "Positive" for s in row_sentiments_predictions) and row_sentiments_predictions: # Ensure list is not empty
                        overall_row_sentiment = "Positive"
                    elif not texts_for_sentiment: # No text found in selected columns
                        overall_row_sentiment = "N/A"
                    
                    row_dict["Overall Sentiment"] = overall_row_sentiment
                    temp_new_rows.append(row_dict)

                if processed_rows_list: # If there were previous rows
                     updated_df = pd.DataFrame(processed_rows_list + temp_new_rows)
                else: # If processing from scratch or no previous rows
                     updated_df = pd.DataFrame(temp_new_rows)
                
                # Ensure original columns + Overall Sentiment are present, in a sensible order
                final_cols = fetched_df_sa.columns.tolist() + ["Overall Sentiment"]
                # Remove duplicates if "Overall Sentiment" was somehow already there
                final_cols = list(dict.fromkeys(final_cols)) 
                st.session_state.sentiment_processed_df = updated_df[final_cols] if not updated_df.empty else pd.DataFrame(columns=final_cols)

                st.session_state.sentiment_display_trigger +=1

                current_sa_settings['sentiment_last_processed_row_index'] = num_total_rows_sa # Update to total rows processed
                save_app_state_from_dict(current_sa_settings)
                
                sentiment_status_placeholder.success(
                    f"Sentiment: Processed {len(new_rows_to_process_df)} rows. Display updated. Last processed index: {current_sa_settings['sentiment_last_processed_row_index']}."
                )
                # For live update, we might want a short wait then rerun, or make it more event driven if possible
                # The current structure uses REFRESH_INTERVAL for subsequent checks.
                should_wait_sa = True # Let it fall through to the refresh interval logic

            else:
                sentiment_status_placeholder.info(
                    f"Sentiment: No new rows found beyond row {start_index_sa}. Next check in {REFRESH_INTERVAL}s."
                )
        elif fetched_df_sa is None and st.session_state.sentiment_analyzer_running:
            sentiment_status_placeholder.error(
                f"Sentiment: Failed to fetch data for Sheet ID: '{current_sa_settings['sentiment_sheet_id']}'. Retry in {REFRESH_INTERVAL}s."
            )
        
        if st.session_state.sentiment_analyzer_running and should_wait_sa:
            time.sleep(REFRESH_INTERVAL)
            st.rerun()

    elif 'app_state_loaded' in st.session_state and not st.session_state.sentiment_analyzer_running:
        sentiment_status_placeholder.info(
            f"Sentiment Analyzer is stopped. Configure and click 'Start'. Last processed for '{st.session_state.run_settings.get('sentiment_sheet_id', 'N/A')}' was row {st.session_state.run_settings.get('sentiment_last_processed_row_index', 'N/A')}."
        )


# ==============================================================================
# TAB 3: AD-HOC SENTIMENT TESTER
# ==============================================================================
with tab3:
    st.header("💬 Ad-hoc Sentiment Tester")
    st.caption("Test the sentiment of any text using the loaded sentiment analysis model (`saved_model.h5`).")

    adhoc_user_text = st.text_area(
        "Enter text for sentiment analysis:",
        height=150,
        key="adhoc_text_input",
        placeholder="E.g., 'The food was amazing and the service was excellent!' or 'I am very disappointed with the quality.'"
    )

    if st.button("🔍 Analyze Sentiment", key="adhoc_analyze_button", type="primary"):
        if adhoc_user_text and adhoc_user_text.strip():
            with st.spinner("Analyzing sentiment..."):
                # The model and artifacts are loaded at app startup via load_sentiment_model_and_artifacts()
                # predict_sentiments_for_texts expects a list of texts
                predictions = predict_sentiments_for_texts([adhoc_user_text.strip()])
                
                if predictions:
                    result = predictions[0] # We sent one item, so we expect one result
                    if result == "Positive":
                        st.success(f"**Sentiment: {result}** 👍")
                    elif result == "Negative":
                        st.error(f"**Sentiment: {result}** 👎")
                    elif result == "N/A":
                        st.warning(f"**Sentiment: {result}** 🤔 \n\n(Could not determine sentiment. The text might be too short, neutral after processing, or contain only out-of-vocabulary words.)")
                    elif "Error: Model not loaded" in result:
                        st.error(f"**Analysis Error:** {result}. The sentiment model could not be loaded. Check console for details.")
                    elif "Error:" in result:
                        st.error(f"**Analysis Error:** {result}")
                    else: 
                        st.info(f"**Result: {result}**")
                else:
                    st.error("Analysis failed to return a result. This might indicate an issue with the sentiment prediction function.")
        else:
            st.warning("Please enter some text to analyze.")
    
    st.markdown("---")
    st.markdown(
        """
        <small>This tool uses a pre-trained sentiment analysis model (based on TensorFlow/Keras) and associated preprocessing artifacts (`saved_model.h5`, `preproc_artifacts.json`) 
        which are loaded when the application starts. Ensure these files are present in the same directory as `sentiment_analyzer.py`.
        </small>
        """,
        unsafe_allow_html=True
    )