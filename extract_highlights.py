# -*- coding: utf-8 -*-
"""
Script to extract highlighted text from images in a directory using the Gemini API,
sort the extracted text based on approximate reading order, and output the results
as a formatted Markdown bullet list.

Handles API key via environment variable (GEMINI_API_KEY) or a .env file.
Accepts configuration via command-line arguments.
"""

import time
import google.generativeai as genai
from PIL import Image, UnidentifiedImageError
import json
import os
import argparse # For command-line arguments

# Attempt to import dotenv for .env file support (optional)
try:
    from dotenv import load_dotenv
    load_dotenv() # Loads variables from .env file if it exists
    print("`.env` file loaded (if found).")
except ImportError:
    print("`python-dotenv` not installed. Relying solely on environment variables.")
    pass # dotenv is optional

# --- Constants ---
DEFAULT_MODEL = "gemini-1.5-flash-latest" # Use a generally available model as default
# DEFAULT_MODEL = "gemini-2.5-pro-exp-03-25" # Or keep the experimental one
DEFAULT_TOLERANCE = 10 # Default pixel tolerance for sorting lines
DEFAULT_SLEEP = 5      # Default seconds between API calls

# --- API Key Setup ---
def configure_genai():
    """Configures the Generative AI client using an environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("="*50)
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set the GEMINI_API_KEY environment variable or create a .env file with:")
        print("GEMINI_API_KEY=YOUR_API_KEY_HERE")
        print("="*50)
        return False
    try:
        genai.configure(api_key=api_key)
        print("GenAI Client configured successfully via environment variable.")
        # Optional: Verify connection by listing models (might incur small cost)
        # genai.list_models()
        return True
    except Exception as e:
        print(f"Error configuring GenAI Client: {e}")
        print("Please ensure your API key is correct and valid.")
        return False

# --- Function Definitions ---

def extract_highlights(image_path, model_name=DEFAULT_MODEL):
    """
    Extracts highlighted text from an image using the GenerativeModel
    in NON-STREAMING mode.

    Args:
        image_path (str): Path to the image file.
        model_name (str): Name of the Gemini model to use.

    Returns:
        tuple: (plain_text_fallback, json_output_list)
               Returns error message in plain_text_fallback on failure.
    """
    try:
        print(f"  Opening image: {os.path.basename(image_path)}")
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"  Error: Image file not found at {image_path}")
        return f"Error: File not found {image_path}", []
    except UnidentifiedImageError:
        print(f"  Error: Cannot identify/open image file: {image_path}")
        return f"Error: Cannot open image {image_path}", []
    except Exception as e:
        print(f"  An unexpected error occurred opening image {image_path}: {e}")
        return f"Error opening image: {e}", []

    # Initialize the GenerativeModel within the function call
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"  Error initializing model '{model_name}': {e}")
        return f"Error initializing model: {e}", []


    # Define the prompt (keeping it consistent)
    prompt = (
        "Analyze the image and extract all text segments that are highlighted "
        "in any shade of yellow or green. For each distinct highlighted segment found, "
        "provide its text content and its 2D bounding box coordinates (box_2d). "
        "Return the results ONLY as a single JSON list, where each item in the list "
        "is an object like: {\"box_2d\": [x_min, y_min, x_max, y_max], \"text\": \"extracted text\"}. "
        "Do not include any other text or explanation outside of this JSON list."
        " Ensure coordinates are integers. Ensure that these coordinates are correct."
    )

    # Define the Generation Config as a Dictionary (using temperature 0 for consistency)
    config_dict = {"temperature": 0.0}

    print(f"  Sending request to model '{model_name}' with config: {config_dict} (Non-Streaming)...")

    # Call generate_content WITHOUT stream=True
    try:
        response = model.generate_content(
            contents=[image, prompt],
            generation_config=config_dict,
            # safety_settings=... # Add safety settings if needed
        )

        # Process the COMPLETE response object
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             block_reason = response.prompt_feedback.block_reason
             print(f"  Warning: Request blocked for image {image_path}. Reason: {block_reason}")
             return f"Error: Request blocked - {block_reason}", []

        if hasattr(response, 'text'):
             raw_response_text = response.text
             print("  Response received.")
        else:
             print(f"  Warning: Received response object without 'text' attribute for {image_path}. Response parts: {response.parts if hasattr(response, 'parts') else 'N/A'}")
             raw_response_text = ""

    except Exception as api_err:
        print(f"  Error during API call for {image_path}: {api_err}")
        # Add specific checks for common API errors if desired (e.g., QuotaExceeded)
        return f"Error during API call: {api_err}", []


    # Parse full response text
    json_output = []
    plain_text_fallback = ""

    cleaned_text = raw_response_text.strip()
    if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
    if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    if cleaned_text:
        print(f"  Attempting JSON parse...")
        try:
            parsed_data = json.loads(cleaned_text)
            if isinstance(parsed_data, list):
                is_valid_structure = all(
                    isinstance(item, dict) and 'box_2d' in item and
                    isinstance(item.get('box_2d'), list) and len(item.get('box_2d', [])) == 4 and
                    all(isinstance(coord, (int, float)) for coord in item.get('box_2d', [])) and
                    ('text' in item or 'text_content' in item)
                    for item in parsed_data)

                if is_valid_structure:
                    print(f"  JSON parsed successfully, normalizing keys...")
                    normalized_boxes = []
                    for item in parsed_data:
                        normalized_item = item.copy()
                        if 'text_content' in normalized_item and 'text' not in normalized_item:
                            normalized_item['text'] = normalized_item.pop('text_content')
                        try:
                            normalized_item['box_2d'] = [int(round(c)) for c in normalized_item.get('box_2d', [])]
                        except (ValueError, TypeError):
                            print(f"    Warning: Could not convert coordinates for box: {normalized_item}. Skipping.")
                            continue
                        if 'text' in normalized_item:
                            normalized_boxes.append(normalized_item)
                        else:
                            print(f"    Warning: Missing 'text' key after normalization: {normalized_item}. Skipping.")
                    json_output = normalized_boxes
                    print(f"  Normalization complete. Found {len(json_output)} valid boxes.")
                else:
                     print(f"  Warning: Parsed JSON list items did not match expected format. Treating as fallback.")
                     plain_text_fallback = cleaned_text
            else:
                print(f"  Warning: Parsed JSON was not a list. Treating as fallback.")
                plain_text_fallback = cleaned_text
        except json.JSONDecodeError as json_err:
            print(f"  Warning: Could not decode response as JSON. Error: {json_err}. Treating as fallback.")
            plain_text_fallback = raw_response_text
    else:
         print(f"  Info: No processable text content received.")

    return plain_text_fallback, json_output


def sort_boxes_with_tolerance(boxes, y_tolerance=DEFAULT_TOLERANCE):
    """
    Sorts boxes based on reading order, grouping boxes on nearly horizontal lines.

    Args:
        boxes (list): A list of box dictionaries.
        y_tolerance (int): Max vertical distance (pixels) between centers for grouping.

    Returns:
        list: A new list of sorted box dictionaries.
    """
    if not boxes: return []
    valid_boxes = []
    for i, box in enumerate(boxes):
        if (isinstance(box, dict) and 'box_2d' in box and
            isinstance(box.get('box_2d'), list) and len(box.get('box_2d', [])) == 4 and
            all(isinstance(coord, int) for coord in box.get('box_2d', [])) and 'text' in box):
                y_min, y_max = box['box_2d'][1], box['box_2d'][3]
                box['_y_center'] = (y_min + y_max) / 2.0
                valid_boxes.append(box)
        else:
             print(f"    Warning: Invalid box format during sorting: {box}. Skipping.")
    if not valid_boxes: return []
    valid_boxes.sort(key=lambda box: box['_y_center'])
    sorted_lines = []
    visited = [False] * len(valid_boxes)
    for i in range(len(valid_boxes)):
        if visited[i]: continue
        visited[i] = True
        current_line = [valid_boxes[i]]
        reference_y_center = valid_boxes[i]['_y_center']
        for j in range(i + 1, len(valid_boxes)):
            if not visited[j]:
                y_center_diff = abs(valid_boxes[j]['_y_center'] - reference_y_center)
                if y_center_diff <= y_tolerance:
                    visited[j] = True
                    current_line.append(valid_boxes[j])
        current_line.sort(key=lambda box: box['box_2d'][0])
        sorted_lines.extend(current_line)
    for box in sorted_lines:
        if '_y_center' in box: del box['_y_center']
    return sorted_lines


def process_images(input_dir, output_file, model_name, tolerance, sleep_duration):
    """
    Processes images in a directory, extracts highlights, sorts results,
    and writes formatted output to file and console.

    Args:
        input_dir (str): Path to the directory containing images.
        output_file (str): Path to the output Markdown file.
        model_name (str): Name of the Gemini model to use.
        tolerance (int): Y-tolerance for sorting.
        sleep_duration (int): Seconds to sleep between API calls.
    """
    print("-" * 30)
    print(f"Starting image processing...")
    print(f"Source Directory: {input_dir}")
    print(f"Output File: {output_file}")
    print(f"Model Name: {model_name}")
    print(f"Y-Sort Tolerance: {tolerance}px")
    print(f"Sleep Duration: {sleep_duration}s")
    print("-" * 30)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return # Exit if input directory is invalid

    try:
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff')
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
        image_files.sort()
    except OSError as e:
        print(f"Error accessing image directory {input_dir}: {e}")
        return
    if not image_files:
        print(f"No image files matching extensions {image_extensions} found in directory: {input_dir}")
        return

    total_images = len(image_files)
    print(f"Found {total_images} images to process.")
    processed_count = 0
    error_count = 0
    start_time_total = time.time()

    try:
        with open(output_file, 'w', encoding='utf-8') as md_file:
            for i, image_file in enumerate(image_files):
                print(f"\n--- Processing image {i+1}/{total_images}: {image_file} ---")
                image_path = os.path.join(input_dir, image_file)
                start_time_image = time.time()

                plain_text, json_boxes_unsorted = extract_highlights(image_path, model_name)

                if "Error:" in plain_text and not json_boxes_unsorted:
                    error_count += 1
                    md_file.write(f"## {image_file}\n\n")
                    md_file.write(f"**Error during extraction:**\n```\n{plain_text}\n```\n\n")
                    print(f"  ERROR during extraction: {plain_text}")
                    print(f"  Finished processing {image_file} with an error.")
                    # Optional: Add sleep even after error to avoid hammering API if error is transient
                    # time.sleep(sleep_duration)
                    continue

                print(f"  Sorting {len(json_boxes_unsorted)} extracted boxes...")
                sorted_json_boxes = sort_boxes_with_tolerance(json_boxes_unsorted, y_tolerance=tolerance)

                # --- Write to Markdown File ---
                md_file.write(f"## {image_file}\n\n")
                if plain_text and not sorted_json_boxes:
                    md_file.write(f"**Fallback Text (JSON issue):**\n```\n{plain_text}\n```\n\n")
                elif sorted_json_boxes:
                    md_file.write("**Highlighted Text:**\n\n")
                    for box in sorted_json_boxes:
                        extracted_text = box.get('text', '').strip()
                        if extracted_text:
                            formatted_text = f"<mark>{extracted_text}</mark>"
                            md_file.write(f"* {formatted_text}\n")
                    md_file.write("\n")
                elif not plain_text:
                     md_file.write("*No highlighted text found or extracted.*\n\n")

                # --- Print Formatted Output to Console ---
                print(f"\n--- Console Output for {image_file} ---")
                if plain_text and not sorted_json_boxes:
                    print(f"**Fallback Text:**\n{plain_text}")
                elif sorted_json_boxes:
                    print("**Highlighted Text:**")
                    for box in sorted_json_boxes:
                        extracted_text = box.get('text', '').strip()
                        if extracted_text:
                            formatted_text = f"<mark>{extracted_text}</mark>"
                            print(f"* {formatted_text}")
                elif not plain_text:
                     print("* No highlighted text found or extracted.")
                print("-" * (len(image_file) + 24)) # Divider line

                processed_count += 1
                end_time_image = time.time()
                print(f"  Finished processing {image_file} in {end_time_image - start_time_image:.2f} seconds.")

                # Avoid sleep for the very last image
                if i < total_images - 1:
                    print(f"  Waiting {sleep_duration}s before next image...")
                    time.sleep(sleep_duration)

    except IOError as e:
        print(f"\nCritical Error: Could not write to output file {output_file}: {e}")
        error_count = total_images - processed_count
    except Exception as e:
        print(f"\nCritical Error: An unexpected error occurred during image processing loop: {e}")
        error_count = total_images - processed_count
    finally:
        end_time_total = time.time()
        total_duration = end_time_total - start_time_total
        print("-" * 30)
        print("Processing Summary:")
        print(f"  Total images found: {total_images}")
        print(f"  Successfully processed: {processed_count}")
        print(f"  Errors encountered: {error_count}")
        print(f"  Total processing time: {total_duration:.2f} seconds")
        if processed_count > 0:
             avg_time = (total_duration - (processed_count -1) * sleep_duration) / processed_count if processed_count > 1 else total_duration
             print(f"  Average processing time per image: {avg_time:.2f} seconds (excluding sleep)")
        print(f"Results saved to: {output_file}")
        print("-" * 30)

# --- Main Execution Block ---
def main():
    """Parses command-line arguments and runs the image processing."""

    # --- Check Dependencies ---
    try:
        import PIL
    except ImportError:
        print("="*50)
        print("Error: Pillow (PIL) library not found.")
        print("Please install dependencies using: pip install -r requirements.txt")
        print("or at least: pip install Pillow google-generativeai")
        print("="*50)
        exit(1)

    # --- Configure API Key ---
    if not configure_genai():
        exit(1) # Exit if API key setup failed

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Extract highlighted text from images in a directory using Gemini API."
    )
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Path to the directory containing the images to process."
    )
    parser.add_argument(
        "-o", "--output-file",
        required=True,
        help="Path to the output Markdown file."
    )
    parser.add_argument(
        "-t", "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help=f"Y-axis tolerance (pixels) for grouping text lines during sorting (default: {DEFAULT_TOLERANCE})."
    )
    parser.add_argument(
        "-s", "--sleep",
        type=int,
        default=DEFAULT_SLEEP,
        help=f"Seconds to sleep between API calls to manage rate limits (default: {DEFAULT_SLEEP})."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Name of the Gemini model to use (default: {DEFAULT_MODEL})."
    )

    args = parser.parse_args()

    # --- Run Processing ---
    process_images(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model_name=args.model,
        tolerance=args.tolerance,
        sleep_duration=args.sleep
    )

if __name__ == "__main__":
    main()