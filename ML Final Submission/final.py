import pytesseract
import requests
import re
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import difflib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Unit Mapping Dictionary
unit_mapping = {
    'centilitre': 'centilitre', 'cl': 'centilitre', 'cL': 'centilitre', 'cl.': 'centilitre', 'cl ': 'centilitre',
    'cubic foot': 'cubic foot', 'ft³': 'cubic foot', 'cu ft': 'cubic foot', 'cf': 'cubic foot', 'cubic foot.': 'cubic foot',
    'cubicfoot': 'cubic foot', 'cu. ft': 'cubic foot', 'cubicfoot.': 'cubic foot', 'cubic-foot': 'cubic foot',
    'cubic inch': 'cubic inch', 'in³': 'cubic inch', 'cu in': 'cubic inch', 'ci': 'cubic inch', 'in': 'cubic inch',
    'cubic inch.': 'cubic inch', 'cubicinch': 'cubic inch', 'cuin': 'cubic inch',
    'cup': 'cup', 'c': 'cup', 'cups': 'cup', 'cup.': 'cup', 'cup ': 'cup',
    'decilitre': 'decilitre', 'dl': 'decilitre', 'dL': 'decilitre', 'deciliter': 'decilitre', 'deciliter.': 'decilitre',
    'decilitre.': 'decilitre', 'dl.': 'decilitre',
    'fluid ounce': 'fluid ounce', 'fl oz': 'fluid ounce', 'oz': 'fluid ounce', 'oz fl': 'fluid ounce', 'fl. oz': 'fluid ounce',
    'floz': 'fluid ounce', 'ozfl': 'fluid ounce', 'fl.oz': 'fluid ounce', 'fluid-ounce': 'fluid ounce',
    'fluidounce': 'fluid ounce', 'oz. fl': 'fluid ounce', 'fl. oz.': 'fluid ounce', 'fl. oz ': 'fluid ounce',
    'gallon': 'gallon', 'gal': 'gallon', 'US gal': 'gallon', 'USG': 'gallon', 'gal.': 'gallon', 'gallon.': 'gallon',
    'gal ': 'gallon', 'USgal': 'gallon', 'USG ': 'gallon',
    'imperial gallon': 'imperial gallon', 'imp gal': 'imperial gallon', 'UK gal': 'imperial gallon', 'IG': 'imperial gallon',
    'imperialgal': 'imperial gallon', 'impgal': 'imperial gallon', 'UKgal': 'imperial gallon',
    'litre': 'litre', 'liter': 'litre', 'l': 'litre', 'L': 'litre', 'lt': 'litre', 'litre.': 'litre', 'liter.': 'litre',
    'l.': 'litre', 'l ': 'litre', 'L ': 'litre',
    'microlitre': 'microlitre', 'µL': 'microlitre', 'uL': 'microlitre', 'mcl': 'microlitre', 'microliter': 'microlitre',
    'microL': 'microlitre', 'µL.': 'microlitre', 'uL.': 'microlitre', 'uL ': 'microlitre',
    'millilitre': 'millilitre', 'ml': 'millilitre', 'mL': 'millilitre', 'cc': 'millilitre', 'mL.': 'millilitre', 'mI': 'millilitre','mI.': 'millilitre','mi': 'millilitre','mi.': 'millilitre','ml.': 'millilitre',
    'milliliter': 'millilitre', 'cc.': 'millilitre', 'mL ': 'millilitre', 'ml.': 'millilitre',
    'pint': 'pint', 'pt': 'pint', 'p': 'pint', 'pint.': 'pint', 'pt.': 'pint', 'pint ': 'pint',
    'quart': 'quart', 'qt': 'quart', 'q': 'quart', 'quart.': 'quart', 'qt.': 'quart', 'quart ': 'quart'
}

# Step 1: Function to correct common OCR misreadings
def correct_ocr_errors(text):
    # Replace common misreadings
    text = re.sub(r'(\d)[oO](\d)', r'\1\2', text)  # Replace 'o' or 'O' in numbers
    text = re.sub(r'(\d)I(\d)', r'\1\2', text)  # Replace 'I' with '1'
    text = re.sub(r'(\d)l(\d)', r'\1\2', text)  # Replace 'l' with '1'
    text = re.sub(r'(\d+)l(\d*)', r'\1\2', text)  # Replace 'l' with '1' in numbers
    text = re.sub(r'\b0\b', 'O', text)  # Replace single '0' with 'O'
    text = re.sub(r'\b1\b', 'I', text)  # Replace single '1' with 'I'
    text = re.sub(r'\b5\b', 'S', text)  # Replace single '5' with 'S'
    text = re.sub(r'\b8\b', 'B', text)  # Replace single '8' with 'B'
    text = re.sub(r'\b(\d{1,2})\s*(\d{2,4})\b', r'\1\2', text)  # Merge broken numbers
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    text = re.sub(r'(\d)\s+([a-zA-Z])', r'\1 \2', text)  # Fix spaces between numbers and letters
    text = re.sub(r'([a-zA-Z])\s+(\d)', r'\1 \2', text)  # Fix spaces between letters and numbers
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Ensure space between numbers and letters
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Ensure space between letters and numbers
    text = re.sub(r'(\d+)\s*([a-zA-Z]+)', r'\1 \2', text)  # Ensure space between numbers and units
    text = re.sub(r'(\d+)\s*([a-zA-Z]+)\s*(\d+)', r'\1 \2 \3', text)  # Handle units with numbers before and after
    text = re.sub(r'\b(\d+)[oO](\d+)\b', r'\1\2', text)  # Replace 'o' or 'O' in numbers
    text = re.sub(r'(\d+)\s*\-\s*(\d+)', r'\1-\2', text)  # Handle hyphenated numbers
    text = re.sub(r'(\d+)\s*\(\s*(\d+)\s*\)', r'\1(\2)', text)  # Handle numbers with parentheses
    text = re.sub(r'(\d+)\s*\n\s*(\d+)', r'\1 \2', text)  # Handle newline between numbers
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)  # Handle period between numbers
    text = re.sub(r'(\d+)\s*([a-zA-Z])', r'\1 \2', text)  # Ensure space between numbers and letters
     # Additional rules based on common OCR errors
    text = re.sub(r'\b(\d+)\s*millilitre(s)?\b', r'\1 millilitre', text, flags=re.IGNORECASE)  # Standardize "millilitre" to "millilitre"
    text = re.sub(r'\b(\d+)\s*milliliter(s)?\b', r'\1 millilitre', text, flags=re.IGNORECASE)  # Convert "milliliter" to "millilitre"
    text = re.sub(r'\b(\d+)\s*ml\b', r'\1 millilitre', text, flags=re.IGNORECASE)  # Convert "ml" to "millilitre"
    text = re.sub(r'\b(\d+)\s*fluid ounce(s)?\b', r'\1 fluid ounce', text, flags=re.IGNORECASE)  # Standardize "fluid ounce" to "fluid ounce"
    text = re.sub(r'\b(\d+)\s*fl oz\b', r'\1 fluid ounce', text, flags=re.IGNORECASE)  # Convert "fl oz" to "fluid ounce"
    text = re.sub(r'\b(\d+)\s*oz\b', r'\1 fluid ounce', text, flags=re.IGNORECASE)  # Convert "oz" to "fluid ounce"
    text = re.sub(r'\b(\d+)\s*cup(s)?\b', r'\1 cup', text, flags=re.IGNORECASE)  # Standardize "cup" to "cup"
    return text


# Step 2: Standardize units in the detected text
def standardize_unit(text):
    for unit, standard in unit_mapping.items():
        text = re.sub(r'\b' + re.escape(unit) + r'\b', standard, text, flags=re.IGNORECASE)
    return text

# Step 3: Download and process image from URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        img = Image.open(BytesIO(response.content))
        img = np.array(img)  # Convert to NumPy array
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from URL: {e}")
        return None

# Step 4: Detect text from the image using PyTesseract
def detect_text_from_image(image_source):
    image = load_image_from_url(image_source)
    if image is not None:
        img_pil = Image.fromarray(image)
        text = pytesseract.image_to_string(img_pil)
        return image, text
    return None, None

# Step 5: Merge multiline text into a single string
import re

def extract_volumes(text):
    # Standardize and correct OCR errors
    standardized_text = standardize_unit(text)
    corrected_text = correct_ocr_errors(standardized_text)

    print(f"Full Text: {text}")
    print(f"Standardized Text: {standardized_text}")
    print(f"Corrected Text: {corrected_text}")

    # Adjusted regex to capture both volume number and unit
    volume_pattern = re.compile(
         r'(\d+\.?\d*)\s*(l|litre|liter|ml|millilitre|milliliter|cc|cubic\s*centimeter|cubic\s*inch)?',
        re.IGNORECASE
    )

    # Find all matching volumes
    volumes = re.findall(volume_pattern, corrected_text)

    # Handle cases where only the number is captured
    def convert_unit(unit):
        unit = unit.lower() if unit else 'millilitre'  # Default to 'millilitre' if unit is missing
        if unit in ['liter', 'litre', 'l']:
            return 'liter'
        elif unit in ['ml', 'millilitre', 'milliliter']:
            return 'millilitre'
        elif unit in ['cc', 'cubic centimeter']:
            return 'cc'
        elif unit in ['cubic inch']:
            return 'cubic inch'
        else:
            return unit

    # Standardize volume units for simple matches, handle missing units
    volumes = [(num, convert_unit(unit)) for num, unit in volumes]

    return volumes, [], None  # Return an empty list for custom_volumes and None for range_match





# Step 7: Compare detected value with entity value from CSV using fuzzy matching
def compare_with_fuzzy_matching(detected_value, entity_value):
    detected_value = detected_value.lower().replace(' ', '')
    entity_value = entity_value.lower().replace(' ', '')

    # Remove units for comparison
    detected_value_number = re.sub(r'[^\d]', '', detected_value)
    entity_value_number = re.sub(r'[^\d]', '', entity_value)

    # Handle unit discrepancies
    for unit in unit_mapping.keys():
        if unit in detected_value:
            detected_value_number += ' ' + unit_mapping[unit]

    for unit in unit_mapping.keys():
        if unit in entity_value:
            entity_value_number += ' ' + unit_mapping[unit]

    similarity = difflib.SequenceMatcher(None, detected_value_number, entity_value_number).ratio()

    # Compare the values considering the units
    exact_match = detected_value == entity_value
    return exact_match or similarity >= 0.8  # Adjusted threshold for better matching

# Step 8: Process image and extract relevant information
def process_image(image_source, entity_value):
    img_rgb, text = detect_text_from_image(image_source)
    if img_rgb is None:
        return None, None, None, False

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

    full_text = merge_multi_line_text(text)
    volumes, custom_volumes, range_match = extract_volumes(text)

    if custom_volumes:
        # Ensure that only numeric values are used
        detected_value = max(custom_volumes, key=lambda x: float(x[1]))  # Use the highest volume
        detected_value = f"{detected_value[1]} {detected_value[2]}"
    elif volumes:
        detected_value = f"{volumes[0][0]} {volumes[0][1]}"
    else:
        detected_value = "No volume detected"  # Set a default value or flag

    print("\nDetected Text from Image:")
    print(full_text)

    print(f"Extracted Value: {detected_value}")

    match = compare_with_fuzzy_matching(detected_value, entity_value)
    return detected_value, full_text, text, match

# Step 9: Calculate F1 Score and Accuracy after processing all images
def calculate_scores(results):
    true_labels = [1 if res['match'] else 0 for res in results]  # 1 for correct matches, 0 for incorrect
    predicted_labels = [1 if res['match'] else 0 for res in results]

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate precision, recall, and F1 score
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return accuracy, precision, recall, f1

# Step 10: Run with the Dataset and Save CSV
import pandas as pd

def run_and_save_results(input_file, output_csv, start_idx=0, end_idx=None, criteria_func=None):
    # Read the Excel file
    df = pd.read_excel(input_file)

    # If no end index is specified, process all rows after the start index
    if end_idx is None:
        end_idx = len(df)

    results = []
    for index, row in df.iterrows():
        if index < start_idx:
            continue
        if index >= end_idx:
            break

        # If a criteria function is provided, use it to filter rows
        if criteria_func and not criteria_func(row):
            continue

        image_source = row['image_link']  # Replace with the appropriate column name
        entity_value = row['entity_value']  # Replace with the appropriate column name
        detected_value, full_text, text, match = process_image(image_source, entity_value)

        results.append({
            'image_source': image_source,
            'entity_value': entity_value,
            'detected_value': detected_value,
            'full_text': full_text,
            'text': text,
            'match': match
        })

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

# Example usage
excel_path = '/content/item_volume.xlsx'  # Replace with your Excel file path
output_csv = 'output10.csv'  # Replace with desired output file path
start_idx = 500  # Adjust start index as needed
end_idx = 1000  # Adjust end index as needed
criteria_func = lambda row: row['entity_value'] is not None  # Example criteria function

run_and_save_results(excel_path, output_csv, start_idx=start_idx, end_idx=end_idx, criteria_func=criteria_func)
