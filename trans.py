import os
import torch
from transformers import pipeline
from PIL import Image

# 1. Environment Setup
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# 2. Initialize Pipeline
# Added 'ignore_mismatched_sizes' to prevent potential loading errors
pipe = pipeline(
    "image-classification", 
    model="circulus/vits-age-gender-detect",
    device=0,         
    dtype=torch.float16 
)

# 3. Predict
image_path = "boy2.jpg"
image = Image.open(image_path).convert("RGB")
results = pipe(image)

# --- Improved Mapping (Lowercase for safety) ---
age_mapping = {
    "baby": "0-2",
    "boy": "3-18",
    "girl": "3-18",
    "man": "19-59",
    "woman": "19-59"
}

# --- Smart Parsing Logic ---
top_result = results[0]
raw_label = top_result['label']
confidence = top_result['score']

# DEBUG: Un-comment the line below if you want to see exactly what the model says
# print(f"DEBUG: Raw label from model is: '{raw_label}'")

if "_" in raw_label:
    # This handles "male_20-29" or "female_more_than_70"
    parts = raw_label.split("_")
    gender_raw = parts[0]
    # Join everything after the first underscore back together for the range
    numerical_range = "-".join(parts[1:]) 
else:
    # This handles single words like "Boy", "female", "man"
    gender_raw = raw_label
    numerical_range = age_mapping.get(gender_raw.lower(), "Unknown")

# Final Standardized Gender Logic
gender_lower = gender_raw.lower()
if gender_lower in ["male", "man", "boy"]:
    gender_clean = "Male"
elif gender_lower in ["female", "woman", "girl"]:
    gender_clean = "Female"
else:
    gender_clean = gender_raw.capitalize()

# Clean up range formatting (e.g., changing 'more-than-70' to '70+')
numerical_range = numerical_range.replace("more-than-", ">")

print(f"--- Standardized Results for {image_path} ---")
print(f"Gender:      {gender_clean}")
print(f"Age Range:   {numerical_range}")
print(f"Confidence:  {confidence:.2%}")