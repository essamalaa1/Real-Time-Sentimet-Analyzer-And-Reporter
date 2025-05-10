# Realtime Customer Insights Processor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The Realtime Customer Insights Processor is a Streamlit web application designed to analyze customer feedback in near real-time. It fetches data from Google Sheets, generates insightful summary reports using Large Language Models (LLMs), and performs granular sentiment analysis on customer reviews.

**Project Goal:** To empower businesses to automatically analyze customer feedback, derive actionable insights quickly, and make data-driven decisions efficiently.

## Features

*   **Google Sheets Integration:** Fetches data directly from public Google Sheets via their CSV export links.
*   **LLM-Powered Batch Reporting:**
    *   Processes reviews in configurable batches.
    *   Utilizes Ollama-compatible LLMs (e.g., LLaMA, DeepSeek) for report generation.
    *   Generates structured Markdown reports including:
        *   Executive Summary
        *   Sentiment Analysis Overview
        *   Key Themes & Insights
        *   Actionable Recommendations
    *   Downloadable PDF versions of reports with custom styling.
*   **Row-Level Sentiment Analysis:**
    *   Uses a custom-trained TensorFlow/Keras model (`saved_model.h5` & `preproc_artifacts.json`) for binary sentiment classification (Positive/Negative).
    *   Analyzes specified text columns from Google Sheets.
    *   Displays results in a live-updating table within the app.
*   **Ad-hoc Sentiment Tester:**
    *   Allows users to input any text for immediate sentiment prediction using the same custom model.
*   **Persistent State:** Saves application settings (Sheet IDs, last processed rows, model choices) locally in `review_processor_state.json` to resume processing.
*   **User-Friendly Interface:** Intuitive UI built with Streamlit, organized into three distinct tabs:
    1.  📋 **Batch Reporter**
    2.  🧐 **Sentiment Analyzer**
    3.  💬 **Ad-hoc Sentiment Test**
*   **Data Preview:** Shows a sample of the data from the Google Sheet before starting processing.
*   **Cost-Effective:** Designed to run locally using free and open-source tools, including local LLMs via Ollama and free tiers for model training (Google Colab) and data storage (Google Drive for CSVs).

## Tech Stack

*   **Backend & UI:** Python, Streamlit
*   **LLM Interaction:** Langchain, Ollama
*   **Sentiment Analysis Model:** TensorFlow/Keras
*   **Text Preprocessing:** NLTK, Pandas
*   **PDF Generation:** WeasyPrint, Markdown2
*   **Data Handling:** Pandas
*   **Configuration:** JSON

## Project Structure
.
├── app.py # Main Streamlit application
├── config.py # Configuration (LLM models, prompts, PDF CSS)
├── core_logic.py # Data fetching, LLM report generation
├── sentiment_analyzer.py # Sentiment model loading, preprocessing, prediction
├── pdf_utils.py # Markdown to PDF conversion utilities
├── state_manager.py # Loading/saving application state
├── saved_model.h5 # Pre-trained Keras sentiment model (REQUIRED)
├── preproc_artifacts.json # Preprocessing artifacts for sentiment model (REQUIRED)
├── requirements.txt # Python dependencies
├── review_processor_state.json # (Generated) Stores persistent app state
└── README.md # This file

## Prerequisites

1.  **Python 3.8+**
2.  **Ollama:** Installed and running. See [ollama.ai](https://ollama.ai/).
    *   Pull desired LLM models (e.g., `ollama pull llama3`). Models are configured in `config.py`.
3.  **Sentiment Model Files:**
    *   `saved_model.h5`
    *   `preproc_artifacts.json`
    *   **These files MUST be placed in the root directory of the project (same level as `sentiment_analyzer.py`).** *(You might need to provide these files or instructions on how to obtain/train them if they are not part of the repo).*
4.  **WeasyPrint System Dependencies:**
    *   **Linux (Debian/Ubuntu):** `sudo apt-get install libpango-1.0-0 libcairo2 libgdk-pixbuf-2.0-0`
    *   **macOS:** `brew install pango cairo gdk-pixbuf`
    *   **Windows:** Refer to WeasyPrint documentation for installing GTK3.
5.  **Google Sheets:**
    *   Your Google Sheets must be publicly accessible via "File > Share > Publish to web" as a CSV.
    *   You'll need the Sheet ID (the long string in the middle of the sheet URL).

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/realtime-customer-insights-processor.git
    cd realtime-customer-insights-processor
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK stopwords:**
    Run Python interpreter and execute:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

5.  **Ensure Sentiment Model Files are present:**
    Place `saved_model.h5` and `preproc_artifacts.json` in the project's root directory if they are not already included.

6.  **Verify Ollama is running and models are pulled.**

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser (usually at `http://localhost:8501`).

2.  **Navigate through the tabs:**
    *   📋 **Batch Reporter:**
        *   Enter your Google Sheet ID.
        *   Configure batch size, select columns for LLM input, and choose an LLM model.
        *   Click "▶️ Start Batch Reporting". Reports will appear as they are generated.
    *   🧐 **Sentiment Analyzer:**
        *   Enter your Google Sheet ID.
        *   Select columns containing text for sentiment analysis.
        *   Click "▶️ Start Sentiment Analysis". A table with sentiment predictions will be displayed and updated.
    *   💬 **Ad-hoc Sentiment Test:**
        *   Enter any text into the text area.
        *   Click "🔍 Analyze Sentiment" to get an instant prediction.

## Configuration (`config.py`)

*   `STATE_FILE`: Path to the state persistence file.
*   `REFRESH_INTERVAL`: How often the app checks for new data (in seconds).
*   `MODEL_OPTIONS`: Dictionary mapping display names to Ollama model tags for the Batch Reporter.
*   `SYSTEM_PROMPT_TEMPLATE`: The master prompt used to instruct the LLM for report generation.
*   `PDF_CSS_STYLE`: CSS styles applied to generated PDF reports.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

Please ensure your code adheres to good practices and includes relevant tests if applicable.

## Future Development (Potential Enhancements)

*   **Deployment Options:** Dockerization for easier deployment.
*   **Advanced NLP:**
    *   Topic modeling for batch reports.
    *   Trend analysis of sentiment and themes over time.
    *   Named Entity Recognition (NER).
*   **Expanded Data Sources:**
    *   Direct database connections (PostgreSQL, MySQL).
    *   CSV/Excel file uploads.
    *   Integration with survey platforms or social media APIs.
*   **Enhanced Reporting:** Interactive dashboards within Streamlit.
*   **User Authentication & RBAC:** For enterprise use.
*   **Scalability Improvements:** For larger datasets and more concurrent users.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details (though a LICENSE.md file isn't explicitly listed, it's standard to include one if you're using a license badge).

## Acknowledgements

*   The Streamlit community for the fantastic web framework.
*   The Ollama team for making local LLM hosting accessible.
*   Open-source libraries like TensorFlow, NLTK, Pandas, WeasyPrint, and Langchain.
*   *(If applicable, add any specific individuals or organizations that supported the project, e.g., for providing the initial idea, data, or volunteer time).*

---

**To make this README even better:**

1.  **Add a Screenshot:** Replace the placeholder URL with an actual compelling screenshot of your application in action. This is very important for a UI-based project.
2.  **`requirements.txt`:** Make sure you have a `requirements.txt` file. You can generate it using `pip freeze > requirements.txt` from your activated virtual environment.
3.  **`LICENSE.md`:** If you use the MIT license badge, create a `LICENSE.md` file in your repository root and paste the standard MIT license text into it.
4.  **Sentiment Model Files:** Be very clear about how users get `saved_model.h5` and `preproc_artifacts.json`. If they are too large for GitHub, provide a download link (e.g., from Google Drive, Hugging Face Hub) or instructions on how to train the model themselves (if you provide the training script and data).
5.  **GIF Demo (Optional but great):** A short animated GIF showing the app's key features can be very engaging.
6.  **Contact/Support (Optional):** If you plan to support it, add a way for users to ask questions or report issues (e.g., link to GitHub Issues).
