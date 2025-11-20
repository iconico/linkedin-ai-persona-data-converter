# LinkedIn AI Persona Data Converter

This project is the subject of a multi-part blog series. You can read about the data generation process in detail here:
- **Blog Post:** [Building a LinkedIn ML Persona: Part 1 - The Data Harvest](https://www.nicowesterdale.com/blog/building-a-linkedin-ml-persona-part-1-the-data-harvest)
- **Join the Discussion:** [Comment on LinkedIn](https://www.linkedin.com/posts/iconico_linkedin-is-training-its-ai-on-your-data-activity-7397258553821913088-7M2r)

---

This script transforms a raw LinkedIn data export into high-quality, structured `.jsonl` datasets, ready for fine-tuning a Large Language Model (LLM) to adopt a specific persona.

It uses a powerful LLM (Google's Gemini 1.5 Flash) as a "depolisher" to generate two distinct types of training data from your polished, final-version texts (like your profile summary, posts, and comments).

## Dataset Types

The script can generate two formats, each designed for a different fine-tuning task.

### 1. Instruction-Tuned Pairs (`instruct`)

This format is for teaching the model how to answer questions in your voice. The script takes your polished text (`output_text`) and has Gemini reverse-engineer the plausible, simple question (`input_text`) that might have generated it.

*   **`input_text`**: "Can you write a professional summary about my career?"
*   **`output_text`**: "For over two decades, I've been at the vanguard of technological innovation..."

### 2. Rewrite Pairs (`rewrite`)

This format is for teaching the model how to perform "style transfer", rewriting a bland text into your specific voice. The script takes your polished text (`output_text`) and has Gemini write a generic, neutral, corporate version of it to serve as the `input_text`.

*   **`input_text`**: "Our department successfully implemented a new data-driven initiative which resulted in significant efficiency gains..."
*   **`output_text`**: "We didn't just 'implement an initiative'; we fundamentally re-architected our entire data pipeline, making it brutally efficient..."

## How to Use

1.  **Get a Google AI API Key:**
    *   Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Create a new API key.

2.  **Set Up Your Environment & Data:**
    *   Create a file named `.env` in this directory.
    *   Add your Google API key to the `.env` file:
        ```
        GOOGLE_API_KEY="your_api_key_here"
        ```
    *   Next, tell the script where your LinkedIn data is. You have two options:
        *   **Option A (Simple):** Create a folder named `linkedin_export` inside this directory and place the unzipped contents of your LinkedIn data export there.
        *   **Option B (Flexible):** Add a `LINKEDIN_EXPORT_PATH` variable to your `.env` file, pointing to wherever you unzipped the data.
            ```
            # Example for Windows (use forward slashes):
            LINKEDIN_EXPORT_PATH="C:/Users/YourUser/Documents/Complete_LinkedInDataExport_..."
            ```

3.  **Configure Script Mode:**
    *   Open `converter.py`.
    *   Set the `DATASET_MODE` variable to `'instruct'`, `'rewrite'`, or `'both'` depending on which datasets you want to generate.

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Script:**
    ```bash
    python converter.py
    ```
    The script will call the Gemini API for each piece of high-quality text, which will take time and incur minor API costs.

6.  **Find the Output:** The script will generate up to three files in this directory:
    *   `training_data_raw.jsonl`: A raw dump of all the sanitized, high-quality text extracted from your export. Useful for review.
    *   `training_data_instruct_v3.jsonl`: The instruction-tuned dataset.
    *   `training_data_rewrite_v1.jsonl`: The style-transfer rewrite dataset.

## Advanced Configuration

You can easily configure the script by editing the constants at the top of `converter.py`.

*   **`CSV_COLUMN_MAP`**: Add new source files or specify different columns to extract text from.
*   **`SHARES_DATE_CUTOFF`**: Ignore old, irrelevant posts by setting a date cutoff.
*   **`DEPOLISHER_MODEL`**: Change the Gemini model used for data generation.
