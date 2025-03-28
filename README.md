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