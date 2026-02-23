# Manual Image Sorter

A keyboard-driven desktop app for quickly sorting images into folders, with an optional AI suggestion engine powered by CLIP.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **One image at a time** — displays each image fullscreen in a dark canvas with its filename
- **Folder buttons** — subfolders of the working directory appear as clickable buttons at the bottom
- **Keyboard shortcuts** — press `1`–`9` to instantly move an image to the corresponding folder
- **CLIP-powered suggestions** — after a few sorts, the AI suggests the best folder based on visual similarity to previously sorted images, with a confidence percentage
- **Learns across sessions** — embeddings and sort history are cached to disk (`.clip_cache.pkl`) so suggestions improve over time without re-processing
- **Lookahead pre-computation** — embeddings for upcoming images are computed in the background while you sort
- **Undo** — `Ctrl+Z` moves the last image back to the source folder
- **Skip** — `S` or `→` sends the current image to the back of the queue
- **Progress bar** — tracks how many images have been sorted vs. remaining
- **Collision handling** — automatically renames files if a destination conflict exists

## Supported Formats

`.jpg` `.jpeg` `.png` `.gif` `.bmp` `.webp` `.tiff` `.tif`

## Requirements

- Python 3.10+
- [Pillow](https://python-pillow.org/) — image loading and display
- [transformers](https://huggingface.co/docs/transformers) + [torch](https://pytorch.org/) — CLIP model (optional, for AI suggestions)

## Installation

```bash
# Clone the repo
git clone https://github.com/BJerrell/ManualImageSorter.git
cd ManualImageSorter

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

> **CLIP is optional.** If `transformers` and `torch` are not installed, the app runs normally without suggestions.

## Usage

```bash
# Open a folder picker dialog
python app.py

# Or pass a folder directly
python app.py "C:\Photos\Unsorted"
```

The working folder should contain:
- Images to sort (directly in the folder)
- Subfolders as sort destinations (created manually beforehand)

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` – `9` | Move image to folder #1–9 |
| `Enter` | Accept the CLIP suggestion |
| `S` or `→` | Skip (send to back of queue) |
| `Ctrl+Z` | Undo last move |

## How CLIP Suggestions Work

On startup, `ClipSuggester` loads `openai/clip-vit-base-patch32` in a background thread and seeds its k-nearest-neighbour pool from images already present in the destination folders (up to 10 per folder). As you sort, each decision is recorded as a training example. The next image is compared against all recorded examples via cosine similarity, and the folder with the highest weighted vote is suggested and highlighted in green.

The model and sort history are cached in `.clip_cache.pkl` inside the working folder so suggestions are available immediately on subsequent runs.

## Project Structure

```
ManualImageSorter/
├── app.py              # Main UI and sorting logic (tkinter)
├── clip_suggester.py   # CLIP embedding + kNN suggestion engine
├── requirements.txt    # Python dependencies
└── .gitignore
```
