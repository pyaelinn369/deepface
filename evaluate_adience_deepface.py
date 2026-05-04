#!/usr/bin/env python3
"""
Evaluate Adience folds using DeepFace for age and gender only.

Produces per-sample `results.csv`, annotated images in `out_dir/annotated/`, and `metrics.json`.

Usage:
    python evaluate_adience_deepface.py --adience_root AdienceGender --fold_file AdienceGender/fold_0_data.txt --out_dir results_deepface

This script prefers aligned crops in `AdienceGender/aligned/<user_id>/landmark_aligned_face.<face_id>.<orig>`
and falls back to original image + bbox crop using (x,y,dx,dy) from the fold file with a 40% margin.
"""
import os
import sys
import csv
import json
import math
import re
import argparse
import logging

try:
    import cv2
except Exception as e:
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from e

import numpy as np

from deepface import DeepFace
import glob
import gc


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Adience using DeepFace (age+gender)")
    p.add_argument("--adience_root", default="AdienceGender", help="Path to Adience root folder")
    p.add_argument("--fold_file", default=None, help="Path to fold file (tab-separated). Defaults to <adience_root>/fold_0_data.txt")
    p.add_argument("--out_dir", default="evaluation_deepface", help="Output directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save_images", action="store_true", help="Save annotated images (disabled by default)")
    p.add_argument("--face_size", type=int, default=64, help="Resize face crops to this size (pixels). Use smaller to reduce memory)")
    p.add_argument("--detector_backend", default="skip", help="DeepFace detector backend (default: skip for cropped faces)")
    p.add_argument("--enforce_detection", action="store_true", help="Pass --enforce_detection to DeepFace.analyze")
    p.add_argument("--align", action="store_true", help="Allow DeepFace to align faces (default: False for cropped faces)")
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


def load_face_images(samples, adience_root, image_map):
    imgs = []
    valid_idx = []
    used_paths = []
    for i, s in enumerate(samples):
        aligned_path = os.path.join(adience_root, 'aligned', s['user_id'], f"landmark_aligned_face.{s['face_id']}.{s['original_image']}")
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
        imgs.append(img)
        valid_idx.append(i)
        used_paths.append(used or "")
        if len(imgs) % 200 == 0:
            logging.info("Prepared %d faces...", len(imgs))
    if len(imgs) == 0:
        return np.zeros((0,)), [], []
    return np.asarray(imgs, dtype=np.uint8), valid_idx, used_paths


def stream_face_batches(samples, adience_root, image_map, batch_size=32, face_size=64):
    """Yield (images_array, valid_idx_list, used_paths_list) for each batch.

    Images are returned as uint8 numpy arrays (BGR). This function performs the same
    aligned-or-bbox cropping as the old loader but only keeps one batch in memory.
    """
    batch_imgs = []
    batch_idx = []
    batch_used = []
    processed = 0
    for i, s in enumerate(samples):
        aligned_path = os.path.join(adience_root, 'aligned', s['user_id'], f"landmark_aligned_face.{s['face_id']}.{s['original_image']}")
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
            # free batch memory
            batch_imgs = []
            batch_idx = []
            batch_used = []
            gc.collect()

    if batch_imgs:
        arr = np.asarray(batch_imgs, dtype=np.uint8)
        yield arr, batch_idx, batch_used


def annotate_and_save(image, out_path, gt_age, gt_gender, pred_age, pred_gender, prob_f):
    label = f"GT:{int(gt_age) if gt_age is not None else 'NA'}/{gt_gender or 'NA'}  PRED:{pred_age:.1f}/{pred_gender}  Pf={prob_f:.2f}"
    img = image.copy()
    # Draw small label strip
    cv2.rectangle(img, (2, 2), (img.shape[1] - 2, 28), (0, 0, 0), thickness=-1)
    cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def compute_metrics(rows):
    age_errors = []
    gender_gt = []
    gender_pred = []
    for r in rows:
        if r['gt_age'] is not None:
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


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def predict_with_deepface(images, batch_size, detector_backend, enforce_detection, align):
    n = len(images)
    results_list = []
    for start in range(0, n, batch_size):
        batch = images[start:start + batch_size]
        try:
            preds = DeepFace.analyze(
                img_path=list(batch),
                actions=['age', 'gender'],
                enforce_detection=enforce_detection,
                detector_backend=detector_backend,
                align=align,
                silent=True,
            )
        except Exception:
            logging.exception("Batch DeepFace.analyze failed; falling back to per-image analysis")
            preds = []
            for img in batch:
                try:
                    p = DeepFace.analyze(
                        img_path=img,
                        actions=['age', 'gender'],
                        enforce_detection=enforce_detection,
                        detector_backend=detector_backend,
                        align=align,
                        silent=True,
                    )
                    preds.append(p)
                except Exception:
                    logging.exception("Failed DeepFace.analyze for one image; skipping")
                    preds.append([])

        # Normalize preds into list-per-image where each item is a list of face dicts
        if isinstance(preds, dict):
            preds = [preds]
        for p in preds:
            if isinstance(p, list):
                results_list.append(p)
            elif isinstance(p, dict):
                results_list.append([p])
            else:
                results_list.append([])
    return results_list


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    adience_root = args.adience_root

    # Find fold files: if user provided one, use it; otherwise process all fold_*.txt files under adience_root
    if args.fold_file:
        fold_files = [args.fold_file]
    else:
        pattern = os.path.join(adience_root, "fold*.txt")
        fold_files = sorted(glob.glob(pattern))
        if not fold_files:
            # fallback to the single common filename
            default_path = os.path.join(adience_root, "fold_0_data.txt")
            if os.path.exists(default_path):
                fold_files = [default_path]

    if not fold_files:
        logging.error("No fold files found under %s. Provide --fold_file or place fold_*.txt files there.", adience_root)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    logging.info("Building image basename map (may take a moment)...")
    image_map = build_basename_map(adience_root)

    # Evaluate each fold separately, saving results under a subfolder per fold
    for fold_path in fold_files:
        if not os.path.exists(fold_path):
            logging.warning("Skipping missing fold file: %s", fold_path)
            continue
        fold_name = os.path.splitext(os.path.basename(fold_path))[0]
        fold_out = os.path.join(args.out_dir, fold_name)
        os.makedirs(fold_out, exist_ok=True)
        annotated_dir = os.path.join(fold_out, "annotated")
        if args.save_images:
            os.makedirs(annotated_dir, exist_ok=True)

        logging.info("Evaluating fold %s -> output: %s", fold_path, fold_out)

        samples = load_adience_fold(fold_path)
        if len(samples) == 0:
            logging.warning("No samples found in %s, skipping.", fold_path)
            continue
        logging.info("Loaded %d samples from %s", len(samples), fold_path)

        logging.info("Preparing & predicting in streaming batches (prefer aligned crops, else bbox crop with 40%% margin)...")

        results = []
        processed_total = 0
        for images_batch, batch_idx, batch_used in stream_face_batches(samples, adience_root, image_map, batch_size=args.batch_size, face_size=args.face_size):
            if images_batch is None or images_batch.size == 0:
                continue
            try:
                preds = DeepFace.analyze(
                    img_path=list(images_batch),
                    actions=['age', 'gender'],
                    enforce_detection=args.enforce_detection,
                    detector_backend=args.detector_backend,
                    align=args.align,
                    silent=True,
                )
            except Exception:
                logging.exception("Batch DeepFace.analyze failed; falling back to per-image analysis")
                preds = []
                for img in images_batch:
                    try:
                        p = DeepFace.analyze(
                            img_path=img,
                            actions=['age', 'gender'],
                            enforce_detection=args.enforce_detection,
                            detector_backend=args.detector_backend,
                            align=args.align,
                            silent=True,
                        )
                        preds.append(p)
                    except Exception:
                        logging.exception("Failed DeepFace.analyze for one image; appending empty result")
                        preds.append([])

            # Normalize preds into list-per-image where each item is a list of face dicts
            if isinstance(preds, dict):
                preds = [preds]
            normalized = []
            for p in preds:
                if isinstance(p, list):
                    normalized.append(p)
                elif isinstance(p, dict):
                    normalized.append([p])
                else:
                    normalized.append([])

            for j, preds_for_image in enumerate(normalized):
                if len(preds_for_image) == 0:
                    continue
                face_pred = preds_for_image[0]
                s = samples[batch_idx[j]]
                pred_age = safe_float(face_pred.get('age'))
                gender_scores = face_pred.get('gender', {}) or {}
                prob_f = gender_scores.get('Woman') or gender_scores.get('F') or gender_scores.get('female') or 0.0
                prob_m = gender_scores.get('Man') or gender_scores.get('M') or gender_scores.get('male') or 0.0
                try:
                    prob_f = float(prob_f)
                except Exception:
                    prob_f = 0.0
                try:
                    prob_m = float(prob_m)
                except Exception:
                    prob_m = 0.0
                if prob_f > 1.0:
                    prob_f = prob_f / 100.0
                if prob_m > 1.0:
                    prob_m = prob_m / 100.0
                pred_gender = 'F' if prob_f > prob_m else 'M'

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
                    'prob_female': prob_f,
                    'prob_male': prob_m,
                }
                results.append(row)

                if args.save_images:
                    try:
                        img_u8 = images_batch[j]
                        fname = f"{s['user_id']}_{s['face_id']}_{os.path.basename(s['original_image'])}"
                        out_path = os.path.join(annotated_dir, fname)
                        annotate_and_save(img_u8, out_path, s['age'], s['gender'], pred_age or 0.0, pred_gender, prob_f)
                    except Exception:
                        logging.exception("Failed to annotate/save image for %s", s)

            processed_total += len(normalized)
            logging.info("Processed %d / %d samples for fold %s", min(processed_total, len(samples)), len(samples), fold_name)

        csv_out = os.path.join(fold_out, "results.csv")
        with open(csv_out, "w", newline='') as fh:
            fieldnames = ['user_id', 'original_image', 'face_id', 'used_path', 'gt_age', 'gt_age_str', 'pred_age', 'age_error', 'gt_gender', 'pred_gender', 'prob_female', 'prob_male']
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        metrics = compute_metrics(results)
        with open(os.path.join(fold_out, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)

        logging.info("Finished fold %s. Results CSV: %s", fold_name, csv_out)
        logging.info("Metrics JSON: %s", os.path.join(fold_out, "metrics.json"))
        if args.save_images:
            logging.info("Annotated images: %s", annotated_dir)


if __name__ == "__main__":
    main()
