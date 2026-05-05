#!/usr/bin/env python3
"""
Evaluate Adience (fold_0_data.txt) using the HuggingFace ViT age-gender model.

- Uses processor + model (no HF pipeline) from `abhilash88/age-gender-prediction`.
- Does NOT generate annotated images.
- Produces per-fold output: `results.csv` and `metrics.json` under `--out_dir`.

Example:
    python3 deepface/evaluate_adience_vit.py --adience_root AdienceGender --out_dir evaluation_vit --device 0
"""
import os
import sys
import csv
import json
import math
import re
import argparse
import logging
import gc

try:
    import cv2
except Exception:
    raise ImportError("OpenCV required: pip install opencv-python")

import numpy as np
from PIL import Image

import torch
from transformers import pipeline

try:
    from transformers import AutoImageProcessor as _ImageProcessor
except Exception:
    from transformers import AutoFeatureExtractor as _ImageProcessor
from transformers import AutoModelForImageClassification as _AutoModelClass


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Adience fold_0 using ViT age-gender model")
    p.add_argument("--adience_root", default="AdienceGender", help="Path to Adience root folder")
    p.add_argument("--fold_file", default=None,
                   help="Path to fold file. Default: <adience_root>/fold_0_data.txt")
    p.add_argument("--out_dir", default="evaluation_vit", help="Output directory")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference")
    p.add_argument("--face_size", type=int, default=64, help="Resize face crops to this size to reduce memory")
    p.add_argument("--device", type=int, default=0, help="CUDA device id (0). Use -1 for CPU.")
    return p.parse_args()


def parse_age_string(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        pass
    nums = re.findall(r"\d+", s)
    if len(nums) == 2:
        try:
            return (float(nums[0]) + float(nums[1])) / 2.0
        except Exception:
            pass
    if len(nums) == 1:
        try:
            return float(nums[0])
        except Exception:
            pass
    mapping = {
        "0-2": 1.0, "3-9": 6.0, "10-19": 15.0, "20-29": 25.0, "30-39": 35.0,
        "40-49": 45.0, "50-59": 55.0, "60-69": 65.0, "70+": 75.0
    }
    return mapping.get(s, None)


def parse_gender_string(g):
    if g is None:
        return None
    s = str(g).strip().lower()
    if s == "":
        return None
    if s in ("m", "male", "man", "boy", "0"):
        return "M"
    if s in ("f", "female", "woman", "girl", "1"):
        return "F"
    if s[0] == 'm':
        return 'M'
    if s[0] == 'f':
        return 'F'
    return None


def load_adience_fold(fold_file):
    samples = []
    with open(fold_file, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for row in reader:
            user_id = row.get('user_id', '').strip()
            orig = row.get('original_image', '').strip()
            face_id = row.get('face_id', '').strip()
            age_str = row.get('age', '').strip()
            gender = row.get('gender', '').strip()
            try:
                x = int(float(row.get('x', 0)))
                y = int(float(row.get('y', 0)))
                dx = int(float(row.get('dx', 0)))
                dy = int(float(row.get('dy', 0)))
            except Exception:
                x = y = dx = dy = 0
            samples.append({
                'user_id': user_id,
                'original_image': orig,
                'face_id': face_id,
                'age_str': age_str,
                'age': parse_age_string(age_str),
                'gender': parse_gender_string(gender),
                'x': x, 'y': y, 'dx': dx, 'dy': dy
            })
    return samples


def build_basename_map(root, exclude_dirs=('aligned', 'pretrained_models')):
    mapping = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for f in filenames:
            mapping.setdefault(f, []).append(os.path.join(dirpath, f))
    return mapping


def stream_face_batches(samples, adience_root, image_map, batch_size=32, face_size=64):
    batch_imgs = []
    batch_idx = []
    batch_used = []
    processed = 0
    for i, s in enumerate(samples):
        aligned_path = os.path.join(adience_root, 'aligned', s['user_id'],
                                    f"landmark_aligned_face.{s['face_id']}.{s['original_image']}")
        used = None
        img = None
        if os.path.exists(aligned_path):
            img = cv2.imread(aligned_path)
            used = aligned_path
            if img is None:
                logging.warning("Failed to read aligned image: %s", aligned_path)
        else:
            candidates = image_map.get(s['original_image'])
            if candidates:
                orig_path = candidates[0]
                img_full = cv2.imread(orig_path)
                if img_full is None:
                    logging.warning("Failed to read original image: %s", orig_path)
                else:
                    x, y, dx, dy = s['x'], s['y'], s['dx'], s['dy']
                    if dx <= 0 or dy <= 0:
                        h, w = img_full.shape[:2]
                        side = min(w, h)
                        cx, cy = w // 2, h // 2
                        x1 = max(0, cx - side // 2)
                        y1 = max(0, cy - side // 2)
                        x2 = min(w, cx + side // 2)
                        y2 = min(h, cy + side // 2)
                    else:
                        margin = int(min(dx, dy) * 0.4)
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(img_full.shape[1], x + dx + margin)
                        y2 = min(img_full.shape[0], y + dy + margin)
                    face = img_full[y1:y2, x1:x2]
                    if face is None or face.size == 0:
                        logging.warning("Invalid crop for %s (orig=%s). Skipping.", s['original_image'], orig_path)
                    else:
                        img = face
                        used = orig_path
        if img is None:
            logging.debug("Skipping sample: %s/%s", s['user_id'], s['original_image'])
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if face_size and face_size > 0:
            try:
                img_resized = cv2.resize(img, (face_size, face_size), interpolation=cv2.INTER_AREA)
            except Exception:
                logging.exception("Failed to resize image for sample %s", s)
                continue
            img_to_use = img_resized
        else:
            img_to_use = img

        batch_imgs.append(img_to_use)
        batch_idx.append(i)
        batch_used.append(used or "")
        processed += 1
        if processed % 200 == 0:
            logging.info("Prepared %d faces...", processed)

        if len(batch_imgs) >= batch_size:
            arr = np.asarray(batch_imgs, dtype=np.uint8)
            yield arr, batch_idx, batch_used
            batch_imgs = []
            batch_idx = []
            batch_used = []
            gc.collect()

    if batch_imgs:
        arr = np.asarray(batch_imgs, dtype=np.uint8)
        yield arr, batch_idx, batch_used


def bgr_to_pil(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def is_male_label(label):
    if not label:
        return False
    L = str(label).lower()
    return any(tok in L for tok in ("male", "man", "boy"))


def is_female_label(label):
    if not label:
        return False
    L = str(label).lower()
    return any(tok in L for tok in ("female", "woman", "girl"))


def parse_vit_label(raw_label):
    """
    Parse labels like "male_20-29" or single words like "boy" into (gender_char, pred_age, age_str).
    pred_age is a float (midpoint) or None.
    """
    if raw_label is None:
        return None, None, None
    rl = str(raw_label).strip()
    if "_" in rl:
        parts = rl.split("_", 1)
        gender_raw = parts[0]
        age_part = parts[1]
    else:
        gender_raw = rl
        age_map = {"baby": "0-2", "boy": "3-18", "girl": "3-18", "man": "19-59", "woman": "19-59"}
        age_part = age_map.get(gender_raw.lower(), "")
    # normalize
    age_part = age_part.replace("more-than-", ">").replace("plus", "+").replace(" ", "")
    # ranged like 20-29
    if "-" in age_part:
        nums = re.findall(r"\d+", age_part)
        if len(nums) >= 2:
            a0 = int(nums[0])
            a1 = int(nums[1])
            pred = (a0 + a1) / 2.0
            return ("M" if gender_raw.lower().startswith("m") else "F"), pred, f"({a0},{a1})"
    # plus or > groups
    if "+" in age_part or age_part.startswith(">"):
        nums = re.findall(r"\d+", age_part)
        if nums:
            a0 = int(nums[0])
            pred = a0 + 15.0
            return ("M" if gender_raw.lower().startswith("m") else "F"), pred, f"({a0},+)"
    nums = re.findall(r"\d+", age_part)
    if len(nums) == 1:
        pred = float(nums[0])
        return ("M" if gender_raw.lower().startswith("m") else "F"), pred, f"({nums[0]})"
    return ("M" if gender_raw.lower().startswith("m") else "F"), None, age_part or None


def compute_metrics(rows):
    age_errors = []
    gender_gt = []
    gender_pred = []
    for r in rows:
        if r['gt_age'] is not None and r['pred_age'] is not None:
            age_errors.append(abs(r['gt_age'] - r['pred_age']))
        if r['gt_gender'] is not None and r['pred_gender'] is not None:
            gender_gt.append(r['gt_gender'])
            gender_pred.append(r['pred_gender'])
    metrics = {}
    if len(age_errors) > 0:
        metrics['age_mae'] = float(np.mean(age_errors))
        metrics['age_rmse'] = float(math.sqrt(np.mean(np.square(age_errors))))
    else:
        metrics['age_mae'] = None
        metrics['age_rmse'] = None
    if len(gender_gt) > 0:
        total = len(gender_gt)
        correct = sum(1 for a, b in zip(gender_gt, gender_pred) if a == b)
        metrics['gender_accuracy'] = float(correct) / total
        tp = sum(1 for g, p in zip(gender_gt, gender_pred) if g == 'F' and p == 'F')
        tn = sum(1 for g, p in zip(gender_gt, gender_pred) if g == 'M' and p == 'M')
        fp = sum(1 for g, p in zip(gender_gt, gender_pred) if g == 'M' and p == 'F')
        fn = sum(1 for g, p in zip(gender_gt, gender_pred) if g == 'F' and p == 'M')
        metrics['gender_confusion'] = {'TP(F->F)': tp, 'TN(M->M)': tn, 'FP(M->F)': fp, 'FN(F->M)': fn}
    else:
        metrics['gender_accuracy'] = None
        metrics['gender_confusion'] = None
    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    adience_root = args.adience_root

    if args.fold_file:
        fold_file = args.fold_file
    else:
        fold_file = os.path.join(adience_root, "fold_0_data.txt")

    if not os.path.exists(fold_file):
        logging.error("Fold file not found: %s", fold_file)
        sys.exit(1)

    # Normalize and create output directories (use absolute paths so user can find them)
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    fold_name = os.path.splitext(os.path.basename(fold_file))[0]
    fold_out = os.path.join(args.out_dir, fold_name)
    os.makedirs(fold_out, exist_ok=True)
    logging.info("Results will be written to: %s", fold_out)

    logging.info("Building image basename map...")
    image_map = build_basename_map(adience_root)

    logging.info("Loading samples from %s", fold_file)
    samples = load_adience_fold(fold_file)
    logging.info("Loaded %d samples", len(samples))

    logging.info("Loading model + processor (this will load weights to device %s)...", args.device)

    # Decide device
    if args.device is None or args.device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load processor (preprocessing), disable center-crop if present
    try:
        processor = _ImageProcessor.from_pretrained("abhilash88/age-gender-prediction")
        if hasattr(processor, "do_center_crop") and processor.do_center_crop:
            processor.do_center_crop = False
            logging.info("Disabled processor.do_center_crop to avoid crop_size validation errors")
    except Exception:
        logging.exception("Failed to load/adjust processor; will try pipeline fallback later")
        processor = None

    # Try to load the repo helper (local model.py or from HF hub) so we can use the custom AgeGenderViTModel
    remote_module = None
    try:
        import importlib
        remote_module = importlib.import_module("model")
        logging.info("Loaded local model helper 'model.py'")
    except Exception:
        try:
            from huggingface_hub import hf_hub_download
            helper_path = hf_hub_download(repo_id="abhilash88/age-gender-prediction", filename="model.py")
            import importlib.util
            spec = importlib.util.spec_from_file_location("remote_model", helper_path)
            remote_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(remote_module)
            logging.info("Downloaded and loaded remote model helper from HF hub")
        except Exception:
            logging.exception("Could not load remote helper 'model.py'; will fall back to AutoModel/pipeline if needed")
            remote_module = None

    # Instantiate the model, preferring the custom class if available
    model = None
    try:
        ModelClass = getattr(remote_module, "AgeGenderViTModel", None) if remote_module is not None else None
        if ModelClass is not None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("abhilash88/age-gender-prediction", trust_remote_code=True)
            model = ModelClass.from_pretrained("abhilash88/age-gender-prediction", config=config, trust_remote_code=True)
        else:
            # fallback to the AutoModel class (may not be ideal for this repo)
            model = _AutoModelClass.from_pretrained("abhilash88/age-gender-prediction", trust_remote_code=True)
        model.eval()
        model.to(device)
        if device.type.startswith("cuda"):
            try:
                model.half()
            except Exception:
                pass
    except Exception:
        logging.exception("Failed to load model class; falling back to HF pipeline approach")
        model = None

    rows = []
    processed = 0
    for images_batch, batch_idx, batch_used in stream_face_batches(samples, adience_root, image_map,
                                                                  batch_size=args.batch_size,
                                                                  face_size=args.face_size):
        if images_batch is None or images_batch.size == 0:
            continue
        pil_imgs = [bgr_to_pil(img) for img in images_batch]

        # If we have the custom model loaded, run direct inference and build a preds_batch
        preds_batch = []
        if model is not None:
            try:
                # Prepare inputs via the processor
                try:
                    inputs = processor(images=pil_imgs, return_tensors="pt")
                except TypeError:
                    inputs = processor(images=pil_imgs, return_tensors="pt", do_center_crop=False)

                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor):
                        v = v.to(device)
                        if device.type.startswith("cuda") and v.is_floating_point():
                            v = v.half()
                        inputs[k] = v

                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits
                if logits is None:
                    preds_batch = [[] for _ in pil_imgs]
                else:
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    logits_cpu = logits.detach().cpu()
                    ages_raw = logits_cpu[:, 0]
                    fem_raw = logits_cpu[:, 1]
                    # If gender head did not apply sigmoid, apply it now
                    try:
                        import torch as _torch
                        if (_torch.max(fem_raw) > 1.01) or (_torch.min(fem_raw) < -0.01):
                            fem_raw = _torch.sigmoid(fem_raw)
                    except Exception:
                        pass

                    ages_np = ages_raw.numpy()
                    fem_np = fem_raw.numpy()

                    for kidx in range(len(pil_imgs)):
                        raw_age = float(ages_np[kidx])
                        if raw_age <= 1.5:
                            pred_age = float(raw_age * 100.0)
                        else:
                            pred_age = float(raw_age)
                        pred_age = max(0.0, min(100.0, pred_age))

                        female_prob = float(fem_np[kidx])
                        male_prob = max(0.0, 1.0 - female_prob)

                        # Directly create rows from model outputs to avoid label parsing mistakes
                        s = samples[batch_idx[kidx]]
                        pred_gender = "F" if female_prob >= 0.5 else "M"
                        row = {
                            'user_id': s['user_id'],
                            'original_image': s['original_image'],
                            'face_id': s['face_id'],
                            'used_path': batch_used[kidx],
                            'gt_age': s['age'],
                            'gt_age_str': s['age_str'],
                            'pred_age': float(pred_age) if pred_age is not None else None,
                            'age_error': (abs(s['age'] - pred_age) if (s['age'] is not None and pred_age is not None) else None),
                            'gt_gender': s['gender'],
                            'pred_gender': pred_gender,
                            'prob_female': float(female_prob),
                            'prob_male': float(male_prob),
                        }
                        rows.append(row)
                        processed += 1
                    # Skip the generic pipeline-style parsing for this batch since rows already appended
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    logging.info("Processed %d / %d", min(processed, len(samples)), len(samples))
                    continue
            except Exception as e:
                logging.exception("Direct model batch inference failed: %s", e)
                preds_batch = []
        else:
            # No direct model available; fall back to pipeline-style inference
            try:
                from transformers import pipeline as _pipeline
                pipe = _pipeline("image-classification", model="abhilash88/age-gender-prediction", device=args.device)
                if processor is not None:
                    try:
                        if hasattr(pipe, 'image_processor'):
                            pipe.image_processor = processor
                        elif hasattr(pipe, 'feature_extractor'):
                            pipe.feature_extractor = processor
                    except Exception:
                        logging.exception("Failed to inject processor into fallback pipeline")
                preds_batch = pipe(pil_imgs, top_k=5)
            except Exception:
                logging.exception("Fallback pipeline inference failed; marking batch empty")
                preds_batch = [[] for _ in pil_imgs]

        for j, pred_for_image in enumerate(preds_batch):
            if pred_for_image is None:
                continue
            if isinstance(pred_for_image, dict):
                pred_list = [pred_for_image]
            else:
                pred_list = pred_for_image

            if len(pred_list) == 0:
                continue

            male_sum = 0.0
            female_sum = 0.0
            for item in pred_list:
                lbl = str(item.get('label', '')).strip()
                score = float(item.get('score', 0.0) or 0.0)
                if is_male_label(lbl):
                    male_sum += score
                elif is_female_label(lbl):
                    female_sum += score

            top_lbl = str(pred_list[0].get('label', '')) if pred_list else ""
            gender_char, pred_age, age_str = parse_vit_label(top_lbl)
            if pred_age is None:
                age_vals = []
                age_scores = []
                for item in pred_list:
                    lbl = item.get('label', '')
                    score = float(item.get('score', 0.0) or 0.0)
                    _, a, _ = parse_vit_label(lbl)
                    if a is not None:
                        age_vals.append(a)
                        age_scores.append(score)
                if age_vals and sum(age_scores) > 0:
                    pred_age = float(np.dot(age_vals, age_scores) / (sum(age_scores)))

            pred_gender = "F" if female_sum > male_sum else "M"

            s = samples[batch_idx[j]]
            row = {
                'user_id': s['user_id'],
                'original_image': s['original_image'],
                'face_id': s['face_id'],
                'used_path': batch_used[j],
                'gt_age': s['age'],
                'gt_age_str': s['age_str'],
                'pred_age': pred_age if pred_age is not None else None,
                'age_error': (abs(s['age'] - pred_age) if (s['age'] is not None and pred_age is not None) else None),
                'gt_gender': s['gender'],
                'pred_gender': pred_gender,
                'prob_female': float(female_sum),
                'prob_male': float(male_sum),
            }
            rows.append(row)
            processed += 1

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        logging.info("Processed %d / %d", min(processed, len(samples)), len(samples))

    # write CSV
    csv_out = os.path.join(fold_out, "results.csv")
    try:
        with open(csv_out, "w", newline='') as fh:
            fieldnames = ['user_id', 'original_image', 'face_id', 'used_path', 'gt_age', 'gt_age_str',
                          'pred_age', 'age_error', 'gt_gender', 'pred_gender', 'prob_female', 'prob_male']
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
        logging.info("Wrote results CSV: %s", csv_out)
    except Exception:
        logging.exception("Failed to write results CSV: %s", csv_out)

    # write metrics
    metrics = compute_metrics(rows)
    metrics_out = os.path.join(fold_out, "metrics.json")
    try:
        with open(metrics_out, "w") as fh:
            json.dump(metrics, fh, indent=2)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
        logging.info("Wrote metrics: %s", metrics_out)
    except Exception:
        logging.exception("Failed to write metrics JSON: %s", metrics_out)

    # Sanity check: report whether files exist
    logging.info("Results exist: %s", os.path.exists(csv_out))
    logging.info("Metrics exist: %s", os.path.exists(metrics_out))


if __name__ == "__main__":
    main()
