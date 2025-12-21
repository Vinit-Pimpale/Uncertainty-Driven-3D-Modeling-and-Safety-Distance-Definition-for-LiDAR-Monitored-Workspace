#!/usr/bin/env python3
import os
import sys
import sys
import time

def progress_bar(current, total, bar_length=40):
    """Simple progress bar."""
    fraction = current / total
    filled = int(bar_length * fraction)
    bar = "█" * filled + "-" * (bar_length - filled)
    print(f"\r[rename_frames] Renaming: |{bar}| {current}/{total}", end="", flush=True)

def rename_files(folder):
    print(f"\n[rename_frames] Target folder: {folder}")

    if not os.path.exists(folder):
        print(f"[rename_frames] ERROR: Folder does not exist: {folder}")
        return

    # Get all PLY files
    files = [f for f in os.listdir(folder) if f.lower().endswith(".ply")]

    if len(files) == 0:
        print("[rename_frames] No PLY files found. Skipping.")
        return

    # Numeric sort based on filename (before .ply)
    try:
        files.sort(key=lambda name: float(os.path.splitext(name)[0]))
    except ValueError:
        print("[rename_frames] WARNING: Could not convert some filenames to float.")
        print("              Falling back to alphabetical sort.")
        files.sort()

    total = len(files)
    print(f"[rename_frames] Found {total} frames. Renaming...")

    # Rename with progress bar
    for i, old_name in enumerate(files):
        new_name = f"frame_{i:04d}.ply"
        old_path = os.path.join(folder, old_name)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        progress_bar(i + 1, total)

    print("\n[rename_frames] Completed.\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_frames.py <folder_path>")
        sys.exit(1)

    rename_files(sys.argv[1])
