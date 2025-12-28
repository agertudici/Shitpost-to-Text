#!/usr/bin/env python3
from pathlib import Path
import shutil

# --- Will easily / quickly reset a folder of test items from ./testbackup to ./test

# --- Configuration ---
SOURCE_DIR = Path.home() / "SPTT/testbackup"
TARGET_DIR = Path.home() / "SPTT/test"

# Remove existing test folder
if TARGET_DIR.exists():
    print(f"Removing existing {TARGET_DIR} ...")
    shutil.rmtree(TARGET_DIR)

# Copy testbackup to test
print(f"Duplicating {SOURCE_DIR} -> {TARGET_DIR} ...")
shutil.copytree(SOURCE_DIR, TARGET_DIR)
print("Done.")
