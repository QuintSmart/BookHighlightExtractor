# Image Highlight Extractor Script

This script processes a directory of images (JPG, PNG, WEBP, etc.), uses the Google Gemini API to extract text segments highlighted in yellow or green, sorts the extracted text by approximate reading order, and saves the results as a formatted Markdown bullet list. It also prints the results for each image to the console as it runs.

## Prerequisites

* Python 3.7 or higher installed.
* Access to the Google Gemini API and a valid API Key.

## Setup

1.  **Clone/Download:**
    * Download the script (`extract_highlights.py` or your chosen name) and the `requirements.txt` file into a project directory.

2.  **Create Virtual Environment (Highly Recommended):**
    * Open your terminal or command prompt, navigate to your project directory.
    * Run:
        ```bash
        python -m venv venv
        ```
    * Activate the environment:
        * **macOS / Linux:** `source venv/bin/activate`
        * **Windows (CMD):** `venv\Scripts\activate`
        * **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
        * *(You should see `(venv)` at the beginning of your terminal prompt)*

3.  **Install Dependencies:**
    * With your virtual environment activated, run:
        ```bash
        pip install -r requirements.txt
        ```
        *(This installs `google-generativeai`, `Pillow`, and optionally `python-dotenv`)*

4.  **Set API Key:**
    * **Option A (Recommended - `.env` file):**
        * Create a file named `.env` (the filename starts with a dot) in your project directory.
        * Add your API key to this file on a single line:
            ```env
            GEMINI_API_KEY=YOUR_API_KEY_HERE
            ```
        * Replace `YOUR_API_KEY_HERE` with your actual key.
        * **Important:** If using Git, add `.env` to your `.gitignore` file to avoid committing your key.
    * **Option B (Environment Variable):**
        * Set the `GEMINI_API_KEY` environment variable in your terminal session *before* running the script. How you do this depends on your operating system:
            * **macOS / Linux:**
                ```bash
                export GEMINI_API_KEY="YOUR_API_KEY_HERE"
                ```
            * **Windows (CMD):**
                ```cmd
                set GEMINI_API_KEY=YOUR_API_KEY_HERE
                ```
            * **Windows (PowerShell):**
                ```powershell
                $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
                ```
        * *Note: This variable might only last for the current terminal session.*

## Usage

Run the script from your terminal (make sure your virtual environment is activated first).

```bash
python extract_highlights.py -i <path_to_images> -o <output_markdown_file> [options]
 ```

### Command-Line Arguments

This section details the arguments you can pass to the script:

* `-i`, `--input-dir` ( **Required**):
    * Path to the directory containing the images you want to process.
    * *Example:* `-i ./my_scans`

* `-o`, `--output-file` ( **Required**):
    * Path where the output Markdown file will be saved.
    * *Example:* `-o report.md`

* `-t`, `--tolerance` (Optional):
    * The vertical pixel tolerance used when grouping text lines for sorting. Affects how strictly the script considers text to be on the same line if the image is slightly skewed.
    * *Default:* `10`
    * *Example:* `-t 15`

* `-s`, `--sleep` (Optional):
    * The number of seconds to pause between processing each image. This helps manage API rate limits.
    * *Default:* `5`
    * *Example:* `-s 7`

* `-m`, `--model` (Optional):
    * The specific Gemini model name to use for the API calls.
    * *Default:* `gemini-1.5-flash-latest` (Check script's `DEFAULT_MODEL` constant if different)
    * *Example:* `-m gemini-1.5-pro-latest`
    * **Find available model names here:** [ai.google.dev/models/gemini](https://ai.google.dev/models/gemini)

---

### Example Command

Here is an example of how to run the script with some options:

```bash
python extract_highlights.py -i ./path/to/my/images -o ./output/highlights_report.md -t 12 -s 5
```


This command will:
* Process images in the `./path/to/my/images` directory.
* Save the formatted Markdown output to `./output/highlights_report.md`.
* Use a sorting tolerance of 12 pixels.
* Wait 5 seconds between each image processing step.
* Use the default Gemini model specified in the script.