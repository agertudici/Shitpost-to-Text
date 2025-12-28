#!/usr/bin/env python3

# ------------------------------
# ShitPost-to-Text
# ------------------------------
# This Version: rewrote step 3 to keep confidence variables needed for step 4

import contextlib
import os
import shutil
import string
import subprocess
from subprocess import run
import sys
import traceback
import re
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict, Counter

import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageChops
import yaml

# ------------------------------
# Step 0 - Logging and Terminal
# ------------------------------
# --- Log Setup ---
LOG_PATH = Path(__file__).parent / "debug_log.txt"

class Tee:
    """Duplicate stdout/stderr to a file and console"""
    def __init__(self, file):
        self.file = file
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        sys.__stdout__.write(data)
    def flush(self):
        self.file.flush()
        sys.__stdout__.flush()

log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = sys.stderr = Tee(log_file)

# --- Catch unhandled exceptions globally ---
def handle_exception(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.exit(1)

sys.excepthook = handle_exception

print(f"[LOG] Starting script. Log file: {LOG_PATH}")

# Open Terminal / Log
if os.environ.get("TERMINAL_OPEN") != "1":
    env = os.environ.copy()
    env["TERMINAL_OPEN"] = "1"
    subprocess.Popen(
        ["alacritty", "-e", "python3", sys.argv[0]],
        env=env
    )
    sys.exit(0)

# ------------------------------
# Step 1 - Load Config
# ------------------------------

CONFIG_PATH = Path(__file__).parent / "Config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --- PATHS ---

SPTT_ROOT = Path(config['paths']['sptt_root']).resolve()
IMG_FOLDER = (SPTT_ROOT / config['paths']['target_dir']).resolve()
TEMP_FOLDER = (SPTT_ROOT / config['paths']['temp_folder']).resolve()
TEMP_FOLDER.mkdir(exist_ok=True)
DICT_PATHS = [Path(p) for p in config['paths'].get('dict_paths', [])]

# --- CORE OPTIONS ---

FALLBACK_COUNTER = config['options']['fallback_counter']
MIN_WORD_LEN = config['options']['min_word_len']

# --- OCR OPTIONS ---

OCR_CONFIG = config['ocr']

ENABLED_PSMS = {
    int(psm): enabled
    for psm, enabled in OCR_CONFIG['enabled_psms'].items()
    if enabled
}

CONFIDENCE_FLOOR = OCR_CONFIG['confidence_floor']
MAX_RESULTS_PER_PSM = OCR_CONFIG['max_results_per_psm']
MAX_TOTAL_RESULTS = OCR_CONFIG['max_total_results']

# --- LLM OPTIONS ---

LLM_MODEL = config['llm_options']['llm_model']
LLM_PROMPTS = {
    'cleanup': config['llm_options']['ocr_cleanup'],
    'tags': config['llm_options']['llm_tagging'],
    'title': config['llm_options']['llm_titleing'],
}

# --- FILENAME FILTERS ---

FILENAME_WHITELIST = set(config['filename_filters']['whitelist'])
FILENAME_BLACKLIST = set(config['filename_filters']['blacklist'])

# ------------------------------
# Step 2 - Preprocessing
# ------------------------------

# Clear temp folder
for f in TEMP_FOLDER.iterdir():
    f.unlink()

# Detect images
for img_path in IMG_FOLDER.glob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4"]:
        continue

    print(f"\nProcessing {img_path} …")
    basename = img_path.stem
    ext = img_path.suffix

    # Frame extraction for GIF/MP4
    if img_path.suffix.lower() in [".gif", ".mp4"]:
        frame_path = TEMP_FOLDER / (img_path.stem + ".png")
        run(["ffmpeg", "-y", "-i", str(img_path), "-frames:v", "1", str(frame_path)])
        ocr_target = frame_path
    else:
        ocr_target = img_path

    # Ensure the image is in a compatible mode for PIL operations (RGB or L)
    img = Image.open(ocr_target)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    img.save(ocr_target)  # overwrite so downstream blocks see RGB/L

    image_variants = {}

    base_img = Image.open(ocr_target)
    if base_img.mode not in ("L", "RGB"):
        base_img = base_img.convert("RGB")

    VARIANT_BUILDERS = {
        "original": lambda img: img.copy(),

        "grayscale": lambda img: img.convert("L"),

        "autocontrast": lambda img: ImageOps.autocontrast(img),

        "resize_2x": lambda img: img.resize(
            (img.width * 2, img.height * 2), resample=Image.BICUBIC
        ),

        "resize_2x_autocontrast": lambda img: ImageOps.autocontrast(
            img.resize((img.width * 2, img.height * 2), resample=Image.BICUBIC)
        ),

        "denoise_median": lambda img: img.filter(ImageFilter.MedianFilter(size=3)),

        "adaptive_threshold": lambda img: img.convert("L").point(
            lambda p: 255 if p > 128 else 0
        ),

        "deskew": lambda img: img.copy(),  # placeholder: implement rotation correction if desired

        "transparency_white_bg": lambda img: (
            Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255)), img)
            if img.mode == "RGBA" else img.copy()
        ),

        "transparency_black_bg": lambda img: (
            Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0)), img)
            if img.mode == "RGBA" else img.copy()
        ),

        "invert": lambda img: ImageOps.invert(img.convert("RGB")),  # optional extra

        "sharpen": lambda img: img.filter(ImageFilter.SHARPEN),  # optional extra

        "threshold": lambda img: img.convert("L").point(lambda p: 255 if p > 128 else 0),  # optional extra
    }

    # Build only enabled variants
    for name, flag in config["ocr"]["enabled_versions"].items():
        if flag <= 0:
            continue

        if name not in VARIANT_BUILDERS:
            raise RuntimeError(f"Enabled OCR variant has no builder: {name}")

        image_variants[name] = VARIANT_BUILDERS[name](base_img)

    print("[PREPROCESS] Built variants:", list(image_variants.keys()))

# ------------------------------
# Step 3 - OCR
# ------------------------------

    # ---------- helpers ----------

    def load_dictionaries(dict_paths, min_word_len):
        words = set()
        for p in dict_paths:
            try:
                with open(p, "r", errors="ignore") as f:
                    for line in f:
                        w = line.strip().lower()
                        if len(w) >= min_word_len and w.isalpha():
                            words.add(w)
            except FileNotFoundError:
                continue
        return words


    def score_text(lines, conf, dictionary):
        """
        Lexical plausibility score.
        Intentionally simple and deterministic.
        """
        text = " ".join(lines).lower()
        tokens = re.findall(r"[a-z]+", text)
        dict_hits = sum(1 for t in tokens if t in dictionary)
        return int(conf) + dict_hits

    def run_ocr(img, psm):
        data = pytesseract.image_to_data(
            img,
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )

        lines = []
        current = []

        for txt in data["text"]:
            if txt.strip():
                current.append(txt.strip())
            else:
                if current:
                    lines.append(" ".join(current))
                    current = []
        if current:
            lines.append(" ".join(current))

        confs = [c for c in data["conf"] if isinstance(c, int) and c >= 0]
        avg_conf = int(sum(confs) / len(confs)) if confs else 0

        return lines, avg_conf

    # ---------- config ingestion ----------

    enabled_versions = config["ocr"]["enabled_versions"]
    enabled_psms_cfg = config["ocr"]["enabled_psms"]

    first_pass_variants = [k for k, v in enabled_versions.items() if v == 1]
    second_pass_variants = [k for k, v in enabled_versions.items() if v == 2]

    ENABLED_PSMS = [psm for psm, enabled in enabled_psms_cfg.items() if enabled]

    dictionary = load_dictionaries(
        config["paths"]["dict_paths"],
        config["options"]["min_word_len"],
    )

    # ---------- variant image alias ----------

    variant_images = {
        name: image_variants[name]
        for name in enabled_versions
        if name in image_variants and enabled_versions[name] > 0
    }

    # ---------- first-pass OCR ----------

    first_pass_results = defaultdict(dict)
    psm_scores = defaultdict(int)

    for variant in first_pass_variants:
        img = variant_images[variant]
        for psm in ENABLED_PSMS:
            lines, conf = run_ocr(img, psm)
            score = score_text(lines, conf, dictionary)

            first_pass_results[variant][psm] = {
                "lines": lines,
                "confidence": score,
            }

            psm_scores[psm] += score

    print(f"[OCR] First-pass PSM scores: {dict(psm_scores)}")

    # ---------- decision ----------

    winning_psm = max(psm_scores, key=psm_scores.get)
    print(f"[OCR] Winning PSM: {winning_psm}")

    # ---------- final outputs ----------

    variants_lines = {}
    variants_confidence = {}

    # reuse first-pass
    for variant in first_pass_variants:
        res = first_pass_results[variant][winning_psm]
        variants_lines[variant] = res["lines"]
        variants_confidence[variant] = res["confidence"]

    # run second-pass once
    for variant in second_pass_variants:
        img = variant_images[variant]
        lines, conf = run_ocr(img, winning_psm)
        score = score_text(lines, conf, dictionary)

        variants_lines[variant] = lines
        variants_confidence[variant] = score

    # ---------- logging ----------

    print("\n----- STEP 3 SUMMARY -----")
    print(f"PSM scores: {dict(psm_scores)}")
    print(f"Winning PSM: {winning_psm}\n")
    for v in variants_lines:
        lines = variants_lines[v]
        conf = variants_confidence[v]
        word_count = sum(len(line.split()) for line in lines)
        print(f"Variant: {v}, # of Lines: {len(lines)}, Words: {word_count}, Confidence: {conf}")
    print(f"\n")

# ------------------------------
# Step 4 - Line Normalization and Deduplication
# ------------------------------

    def line_similarity(a, b):
        """Return similarity ratio between two strings"""
        return SequenceMatcher(None, a, b).ratio()

    def collapse_words(lines_group):
        """Collapse words across variants for a single line group"""
        # split each line into words
        split_words = [line.split() for line in lines_group]
        # pad shorter lines
        max_len = max(len(w) for w in split_words)
        for w in split_words:
            while len(w) < max_len:
                w.append("")
        # collapse words by most common word at each position
        collapsed = []
        for words_at_pos in zip(*split_words):
            counts = Counter(words_at_pos)
            # remove empty strings from counts
            if "" in counts:
                del counts[""]
            if counts:
                collapsed.append(counts.most_common(1)[0][0])
        return " ".join(collapsed)

    # Step 1: pick a reference variant (highest confidence preferred)
    ref_variant = max(variants_confidence, key=lambda v: variants_confidence[v])
    ref_lines = variants_lines[ref_variant]
    num_lines = len(ref_lines)

    # Step 2: align all variants line-by-line to reference
    aligned_lines = [[] for _ in range(num_lines)]
    for variant, lines in variants_lines.items():
        for i, ref_line in enumerate(ref_lines):
            # find the closest line in this variant
            if lines:
                best_match = max(lines, key=lambda l: line_similarity(l, ref_line))
            else:
                best_match = ""
            aligned_lines[i].append(best_match)

    # Step 3: collapse words across aligned lines
    merged_lines = []
    for line_group in aligned_lines:
        merged_line = collapse_words(line_group)
        merged_lines.append(merged_line)

    # Logging
    print("----- STEP 4 MERGED LINES -----")
    for idx, line in enumerate(merged_lines, 1):
        print(f"{idx} {line}")
    print("-------------------------------\n")

# ------------------------------
# Step 5 - LLM Cleanup + Tagging
# ------------------------------

    llm_cleaned_text = ""
    llm_tags = "auditreq"

    # Join the normalized / fused lines into a single string for the LLM
    ocr_text_for_llm = " ".join(merged_lines)

    if merged_lines and sum(len(l.split()) for l in merged_lines) >= MIN_WORD_LEN:

        # --- LLM Cleanup ---
        llm_prompt = (
            f"{LLM_PROMPTS['cleanup']}\n"
            "IF YOU'RE NOT CONFIDENT IN YOUR INTERPRETATION, RETURN ONLY 'auditreq'\n"
            f"Text: '{ocr_text_for_llm}'"
        )

        try:
            result = subprocess.run(
                ["ollama", "run", LLM_MODEL, llm_prompt],
                capture_output=True,
                text=True,
                check=True
            )
            llm_cleaned_text = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            llm_cleaned_text = f"LLM call failed: {e}"

        print("----- LLM CLEANED TEXT BEGIN -----")
        print(llm_cleaned_text[:1000])
        print("----- LLM CLEANED TEXT END -------\n")

        # --- Tagging (only if cleanup succeeded) ---
        if llm_cleaned_text.lower() != "auditreq":
            tag_prompt = f"""
    {LLM_PROMPTS['tags']}
    Text: '{llm_cleaned_text}'
    """.strip()

            try:
                result = subprocess.run(
                    ["ollama", "run", LLM_MODEL, "do not return anything (not even conjunctions) other than 3-5 comma-separated single-word tags related to this text. Here are some examples:", tag_prompt],
                    capture_output=True,
                    text=True,
                    check=True
                )
                llm_tags = result.stdout.strip()
            except subprocess.CalledProcessError as e:
                llm_tags = f"LLM tagging failed: {e}"

    else:
        print("OCR text not usable for LLM cleanup (low confidence or empty).")

    print("----- LLM TAGS BEGIN -----")
    print(llm_tags)
    print("----- LLM TAGS END -------\n")

# ------------------------------
# Step 6 - File Naming
# ------------------------------

    def sanitize_name(name, lowercase=True, collapse_underscores=True, trim=True):
        if lowercase:
            name = name.lower()
        name = re.sub(r'[^a-z0-9]', '_', name)
        if collapse_underscores:
            name = re.sub(r'_+', '_', name)
        if trim:
            name = name.strip('_')
        return name

    def meaningful_words_in_name(name, dict_paths=DICT_PATHS, min_len=MIN_WORD_LEN, whitelist=None, blacklist=None):
        # lowercase and remove blacklisted terms
        name_lower = name.lower()
        if blacklist:
            for b in blacklist:
                name_lower = name_lower.replace(b.lower(), '')

        words = re.split(r'[^a-zA-Z]+', name_lower)
        meaningful = []

        # Build combined dictionary set
        dict_words = set()
        for dp in dict_paths:
            if dp.exists():
                with open(dp) as f:
                    dict_words.update(
                        w.strip().lower()
                        for w in f
                        if len(w.strip()) >= min_len
                    )

        # Merge whitelist into dictionary words
        if whitelist:
            dict_words.update(w.lower() for w in whitelist)

        for w in words:
            if len(w) >= min_len and w in dict_words:
                meaningful.append(w)

        return meaningful

    # Detect meaningful words only now, at the point of naming
    llm_title = ""
    dict_words_in_name = meaningful_words_in_name(
        basename,
        dict_paths=DICT_PATHS,
        min_len=MIN_WORD_LEN,
        whitelist=config['filename_filters']['whitelist'],
        blacklist=config['filename_filters']['blacklist']
    )

    if (
        not dict_words_in_name
        and llm_cleaned_text
        and llm_cleaned_text.lower() != "auditreq"
    ):

        title_prompt = f"""
    {LLM_PROMPTS['title']}
    Text: '{llm_cleaned_text}'
    """.strip()

        try:
            result = subprocess.run(
                ["ollama", "run", LLM_MODEL, title_prompt],
                capture_output=True,
                text=True,
                check=True
            )
            llm_title = result.stdout.strip()
        except subprocess.CalledProcessError:
            llm_title = ""

        # --- guards ---
        if llm_title.strip().lower() == "auditreq":
            llm_title = ""

        if not re.search(r"[a-zA-Z0-9]", llm_title):
            llm_title = ""

        print("----- LLM TITLE BEGIN -----")
        print(llm_title)
        print("----- LLM TITLE END -------\n")

    # Decide final filename
    if dict_words_in_name:
        # Case A: meaningful original filename
        final_name = sanitize_name(basename)
        name_source = "original"
    elif llm_title:
        # Case B: LLM-generated title
        final_name = sanitize_name(llm_title)[:50]
        name_source = "llm"
    else:
        # Case C: deterministic fallback
        final_name = ''.join(re.findall(r'[a-zA-Z0-9]', basename))[:10].lower()
        name_source = "fallback"

    # Ensure uniqueness
    counter = 1
    final_path = img_path.parent / f"{final_name}{ext}"
    while final_path.exists():
        final_path = img_path.parent / f"{final_name}_{counter}{ext}"
        counter += 1

    img_path.rename(final_path)
    print(f"Final filename: {final_path.name} (source: {name_source})")

# ------------------------------
# Step 7 - Metadata
# ------------------------------

    if final_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:  # only for image types

        # Prepare values; fall back to empty string if unavailable
        xmp_title = llm_title or ""
        xmp_description = llm_cleaned_text if llm_cleaned_text.lower() != "auditreq" else ""
        xmp_tags = llm_tags if llm_tags.lower() != "auditreq" else ""

        # Convert comma-separated tags to array format and sanitize
        tag_args = []
        if xmp_tags:
            # Split, strip, lowercase, remove punctuation
            tag_list = [
                t.strip().lower().translate(str.maketrans('', '', string.punctuation))
                for t in xmp_tags.split(",")
                if t.strip()
            ]
            # Optional: remove filler words
            filler = {"and", "or", "the", "a", "an"}
            tag_list = [t for t in tag_list if t not in filler]

            for t in tag_list:
                tag_args.extend(["-XMP-dc:Subject+=" + t])

        # Build ExifTool command
        llm_stage = "ocrllm" if llm_cleaned_text.lower() != "auditreq" else "auditreq"

        cmd = [
            "exiftool",
            "-overwrite_original",
            f"-XMP-dc:Title={xmp_title}",
            f"-XMP-dc:Description={xmp_description}"
        ] + tag_args + [str(final_path)]

        try:
            subprocess.run(cmd, check=True)
            print(f"Metadata written to {final_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"ExifTool metadata write failed: {e}")

# ------------------------------
# Step 8 - Cleanup
# ------------------------------

# Delete temp files
shutil.rmtree(TEMP_FOLDER)

# --- HOLD TERMINAL OPEN ---
input("Press Enter to exit…")
