import os
try:
    import torch
except Exception:
    import sys
    sys.stderr.write("Torch is not installed or cannot be imported.\n")
    sys.stderr.write("Please activate the environment with PyTorch or install it (e.g. `pip install torch torchvision`).\n")
    raise
from transformers import pipeline
try:
    from transformers import AutoImageProcessor as _ImageProcessor
except Exception:
    from transformers import AutoFeatureExtractor as _ImageProcessor
from transformers import AutoModelForImageClassification as _AutoModelClass
from PIL import Image
import json
import logging
import sys
import re

# 1. Environment Setup
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# 2. Try to use helper functions from the model repo (preferred)
# prefer `predict_age_gender` / `simple_predict` from the model; fall back to manual inference below
image_path = "teenboy.jpg"
image = Image.open(image_path).convert("RGB")

def load_remote_helper():
    # Try local import first (if the user cloned the repo)
    try:
        import importlib
        remote = importlib.import_module("model")
        predict_age_gender = getattr(remote, "predict_age_gender", None)
        simple_predict = getattr(remote, "simple_predict", None)
        return predict_age_gender, simple_predict, remote
    except Exception:
        pass
    # Try to download the helper file from the HF Hub and import it
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None, None, None
    try:
        helper_path = hf_hub_download(repo_id="abhilash88/age-gender-prediction", filename="model.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("remote_model", helper_path)
        remote = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(remote)
        predict_age_gender = getattr(remote, "predict_age_gender", None)
        simple_predict = getattr(remote, "simple_predict", None)
        return predict_age_gender, simple_predict, remote
    except Exception:
        logging.exception("Failed to download/import helper from hub")
        return None, None, None

predict_age_gender, simple_predict, remote_module = load_remote_helper()

# If we successfully downloaded the repo helper, prefer instantiating its custom model class
if remote_module is not None:
    try:
        from transformers import AutoConfig, AutoImageProcessor

        # Prefer the custom AgeGenderViTModel defined in the remote `model.py`
        ModelClass = getattr(remote_module, "AgeGenderViTModel", None)
        config = AutoConfig.from_pretrained("abhilash88/age-gender-prediction", trust_remote_code=True)
        if ModelClass is not None:
            model = ModelClass.from_pretrained("abhilash88/age-gender-prediction", config=config, trust_remote_code=True)
        else:
            # fallback to a standard auto model if class not present
            model = _AutoModelClass.from_pretrained("abhilash88/age-gender-prediction", trust_remote_code=True)

        processor = AutoImageProcessor.from_pretrained("abhilash88/age-gender-prediction")
        # disable center crop to avoid needing crop_size
        if hasattr(processor, "do_center_crop"):
            processor.do_center_crop = False

        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if device.type == "cuda":
            try:
                model.half()
            except Exception:
                pass

        # prepare inputs and move to device
        inputs = processor(images=image, return_tensors="pt")
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if device.type == "cuda" and v.is_floating_point():
                    v = v.half()
                inputs[k] = v

        with torch.no_grad():
            out = model(**inputs)

        # model returns logits where [0]=age, [1]=female_prob (per helper implementation)
        logits = out.logits[0] if out.logits.dim() > 1 else out.logits
        raw_age = float(logits[0].item())
        # Heuristic: some models return normalized age in [0,1]. If that's the
        # case (raw_age <= 1.5), scale to 0-100. Otherwise assume raw_age is years.
        if raw_age <= 1.5:
            age = int(round(raw_age * 100.0))
        else:
            age = int(round(raw_age))
        age = max(0, min(100, age))
        gender_prob_female = float(logits[1].item())
        gender_prob_male = 1.0 - gender_prob_female
        # Predicted gender and confidence
        if gender_prob_female >= 0.5:
            gender = "Female"
            gender_confidence = gender_prob_female
        else:
            gender = "Male"
            gender_confidence = gender_prob_male

        result = {
            "age": age,
            "gender": gender,
            "gender_confidence": float(gender_confidence),
            "gender_probability_male": float(gender_prob_male),
            "gender_probability_female": float(gender_prob_female),
            "label": f"{age} years, {gender}",
            "score": float(gender_confidence)
        }

        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)
    except Exception:
        logging.exception("Remote model helper path failed; falling back to generic pipeline inference")
        # fall through to generic fallback

# Fallback: initialize processor + model (avoid pipeline call-time preprocessing kwargs)
processor = _ImageProcessor.from_pretrained("abhilash88/age-gender-prediction")
model = _AutoModelClass.from_pretrained("abhilash88/age-gender-prediction")

# move model to device and use half precision on CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda":
    try:
        model.half()
    except Exception:
        pass

# 3. Predict (fallback)
# Prepare inputs with center-crop disabled if possible to avoid crop_size requirement
try:
    if hasattr(processor, "do_center_crop"):
        processor.do_center_crop = False
        inputs = processor(images=image, return_tensors="pt")
    else:
        try:
            inputs = processor(images=image, return_tensors="pt", do_center_crop=False)
        except TypeError:
            inputs = processor(images=image, return_tensors="pt")
except Exception:
    logging.exception("Processor failed to prepare inputs")
    raise
# move tensors to device
for k, v in list(inputs.items()):
    if isinstance(v, torch.Tensor):
        v = v.to(device)
        if device.type == "cuda" and v.is_floating_point():
            v = v.half()
        inputs[k] = v

with torch.no_grad():
    out = model(**inputs)
    logits = out.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=min(5, probs.size(-1)), dim=-1)

results = []
for idx, score in zip(topk.indices[0], topk.values[0]):
    lbl = model.config.id2label[int(idx.item())]
    results.append({"label": lbl, "score": float(score.item())})

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

# Build a JSON-compatible result matching the requested schema
age_value = None
# Prefer extracting a concrete number from the raw label first
m = re.search(r"(\d{1,3})", raw_label)
if m:
    try:
        age_value = int(m.group(1))
    except Exception:
        age_value = None
else:
    nums = re.findall(r"(\d{1,3})", numerical_range)
    if len(nums) == 1:
        age_value = int(nums[0])
    elif len(nums) >= 2:
        a = int(nums[0]); b = int(nums[1])
        age_value = int(round((a + b) / 2.0))

if age_value is None:
    label_text = f"{numerical_range}, {gender_clean}"
else:
    label_text = f"{age_value} years, {gender_clean}"

if gender_clean.lower() in ["female", "woman", "girl"]:
    gender_prob_female = float(confidence)
    gender_prob_male = 1.0 - gender_prob_female
else:
    gender_prob_male = float(confidence)
    gender_prob_female = 1.0 - gender_prob_male

gender_confidence = max(gender_prob_female, gender_prob_male)

result = {
    "age": None if age_value is None else int(age_value),
    "gender": gender_clean,
    "gender_confidence": float(gender_confidence),
    "gender_probability_male": float(gender_prob_male),
    "gender_probability_female": float(gender_prob_female),
    "label": label_text,
    "score": float(gender_confidence)
}

print(json.dumps(result, ensure_ascii=False))