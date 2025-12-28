# ShitPost-to-Text (SPTT)

This script converts OCR text from social media posts, memes, and internet content into clean, readable text suitable for screen readers. It handles preprocessing, OCR, deduplication, LLM cleanup, tagging, file renaming, and metadata embedding. Basically it'll semi-intelligently OCR your folder full of 10 year old screenshots of shitposts. Semi configurable if you wanna try to tweak it to a specific use-case. I can't believe I spent three weeks vibe coding this but it exists now. Do whatever you want with it.

---

## Table of Contents

1. [Core Options](#core-options)
2. [OCR Options](#ocr-options)
3. [LLM Options](#llm-options)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [File Naming and Metadata](#file-naming-and-metadata)

---

## Core Options

* **Paths**

  * `sptt_root`: Base folder for the project.
  * `target_dir`: Folder containing input images/videos.
  * `temp_folder`: Temporary processing folder.
  * `dict_paths`: Word dictionaries for OCR scoring.

* **Extensions**

  * Supported image/video types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.mp4`.

* **Options**

  * `fallback_counter`: Used for filename uniqueness.
  * `min_word_len`: Minimum length for words considered valid.

* **Filename Filters**

  * `whitelist`: Terms that should always be considered valid.
  * `blacklist`: Terms that should be ignored when detecting meaningful filenames.

---

## OCR Options

* **PSM Settings**

  * Multiple page segmentation modes enabled (1, 3, 4, 6, 11) to detect text layouts reliably.
  * Disabled modes are ignored to save processing time.

* **Image Variants**

  * First-pass: `original`, `grayscale`, `autocontrast`
  * Second-pass: `resize_2x`, `resize_2x_autocontrast`, `denoise_median`, `deskew`, `transparency_white_bg`
  * Disabled variants are ignored.

* **Scoring**

  * OCR confidence is combined with dictionary match counts to select the winning PSM.
  * `confidence_floor`: Minimum confidence for passing OCR results.
  * `max_results_per_psm` and `max_total_results` limit the number of OCR outputs.

* **Metadata Markup**

  * Currently deprecated; previously used to extract likes, shares, author handles, platforms, and timestamps.

---

## LLM Options

* **Model**

  * `gemma2:2b`

* **OCR Cleanup Prompt**

  * Focuses on body text.
  * Removes social media background data (timestamps, likes, reposts).
  * Corrects OCR mistakes and normalizes punctuation/whitespace.
  * Produces coherent sentences suitable for screen readers.

* **Tagging**

  * Reference tags include humor, meme, satire, opinion, reaction, story, and many others.

* **Titleing**

  * Returns a 3–5 word title summarizing the text content.
  * No punctuation or additional text.

---

## Usage

1. Place your images/videos in the `target_dir`.
2. Run the script:

```bash
python3 SPTT.py
```

3. The script will:

   * Preprocess images.
   * Run OCR across multiple variants and PSMs.
   * Align and merge text lines.
   * Clean text via the LLM.
   * Tag content and generate a title.
   * Rename files based on meaningful words or LLM output.
   * Embed metadata into images (title, description, tags).

---

## Dependencies

* Python 3
* [Pillow (PIL)](https://pillow.readthedocs.io/)
* [pytesseract](https://pypi.org/project/pytesseract/)
* [yaml](https://pyyaml.org/)
* `ffmpeg` for GIF/MP4 frame extraction
* `exiftool` for embedding metadata
* `ollama` for LLM calls

Install via pip:

```bash
pip install pillow pytesseract pyyaml
```

System dependencies (Linux/macOS):

```bash
sudo apt install tesseract-ocr ffmpeg exiftool
```

---

## File Naming and Metadata

* Filenames are sanitized, lowercased, and stripped of non-alphanumeric characters.
* If the filename contains meaningful words, they are retained; otherwise, the LLM generates a descriptive title.
* Metadata embedded includes:

  * `Title`: LLM-generated or original filename.
  * `Description`: Cleaned OCR text.
  * `Tags`: LLM-generated tags, filtered for common filler words.
