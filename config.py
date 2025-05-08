import os

STATE_FILE = "review_processor_state.json"
REFRESH_INTERVAL = 30
MODEL_OPTIONS = {
    "DeepSeek R1 (1.5B)": "deepseek-r1:1.5b",
    "DeepSeek R1 (8B)": "deepseek-r1:8b",
    "LLaMA 3.2 (1B)": "llama3.2:1b",
    "LLaMA 3 (8b)": "llama3:latest",
    "LLaMA 3.2 (3.2b)": "llama3.2:latest"
}
SYSTEM_PROMPT_TEMPLATE = """You are an expert data analyst for cafes and restaurants, skilled at turning customer feedback into actionable insights.
Your task is to generate a **clear, insightful, and actionable** report for each batch of customer reviews.
The report **must** be in **strict Markdown format** and **must always include all 4 sections** outlined below.

### Report for Batch {batch_range}

**1. Executive Summary**

*   Provide a **concise overview** (3-4 sentences) of the batch.
*   Highlight the overall sentiment, 1-2 most prominent themes (positive or negative), and one key recommendation.
*   Focus on the main takeaways for a busy manager.

**2. Sentiment Analysis**

*   **Overall Sentiment:** [Positive/Neutral/Negative/Mixed] - State the dominant sentiment.
*   **Justification & Score (Inferred):** Briefly explain (1-2 sentences) *why* this sentiment was chosen, referencing review tone. Provide an inferred score (e.g., 4.2/5 or 80% Positive). If no explicit ratings, use a qualitative strength (e.g., 'Strongly Positive').
*   **Key Emotion Drivers (Optional):** List 1-2 primary emotions observed (e.g., "Appreciation for service," "Frustration with wait times").

**3. Key Themes & Insights**

*   Identify 2-3 **significant key themes** from the reviews.
*   For each theme:
    *   **Theme [Number]: [Clear, Descriptive Title]**
    *   **Details:** Describe the theme, what customers are saying, and its impact on their experience.
    *   **Evidence (Optional but good):** Briefly mention if it's a common point or include a very short, anonymized quote/paraphrase.

**4. Actionable Recommendations**

*   Provide 2-3 **practical recommendations** directly linked to the identified themes.
*   For each recommendation:
    *   **Recommendation [Number]: [Specific Action]**
    *   **Rationale:** Briefly explain why this action is suggested, connecting it to a theme, and what positive outcome is expected.

⚠️ **Important Notes:**
*   Every section is mandatory. Do not skip any.
*   If review data is sparse, provide generalized insights and clearly state this limitation.
*   Output must be **only** the Markdown report, starting with "### Report for Batch...". No extra conversation.
"""

PDF_CSS_STYLE = """
@page {
    size: A4;
    margin: 1.5cm; /* Standard margins for A4 */
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #666;
    }
}
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; /* Common, clean sans-serif fonts */
    line-height: 1.6;
    color: #333333;
    font-size: 10pt; /* Base font size */
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Georgia', 'Times New Roman', Times, serif; /* A slightly more formal serif for headings */
    color: #1a1a1a;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    line-height: 1.3;
    page-break-after: avoid; /* Try to keep headings with following content */
}
h1 { /* Report Title */
    font-size: 20pt;
    text-align: center;
    border-bottom: 2px solid #0056b3; /* Accent color */
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 1em;
    color: #0056b3;
}
h2 { /* Main Sections like "1. Executive Summary" */
    font-size: 16pt;
    border-bottom: 1px solid #cccccc;
    padding-bottom: 0.2em;
    margin-top: 1.5em;
}
h3 { /* Sub-sections or Key Themes titles */
    font-size: 12pt;
    color: #2a7aaf; /* Slightly lighter blue */
    margin-top: 1.3em;
}
p {
    margin-bottom: 0.8em;
    text-align: justify; /* Justify paragraphs for a more formal look */
}
ul, ol {
    margin-bottom: 0.8em;
    padding-left: 25px;
}
li {
    margin-bottom: 0.25em;
}
strong, b {
    font-weight: bold;
}
em, i {
    font-style: italic;
}
pre, code { /* For any code-like output, though less likely in these reports */
    font-family: 'Menlo', 'Consolas', 'Courier New', Courier, monospace;
    background-color: #f5f5f5;
    padding: 0.1em 0.3em;
    border-radius: 3px;
    font-size: 0.9em;
    border: 1px solid #eeeeee;
    white-space: pre-wrap; /* Allow wrapping */
    word-wrap: break-word;
}
pre {
    padding: 0.8em;
    overflow-x: auto;
}
.report-content {
    /* This class is added around the main HTML body in the template */
}
/* Specific styling for items like Sentiment Score */
.sentiment-analysis p:first-of-type { /* Assuming "Sentiment: Positive" */
    font-weight: bold;
}
.sentiment-analysis p:nth-of-type(2) { /* Assuming "Score: 4.2/5" */
    font-style: italic;
    color: #555;
}
"""