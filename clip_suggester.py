"""
CLIP-based folder suggestion engine.

Loads openai/clip-vit-base-patch32 in a background thread, computes
image embeddings (cached to disk), and uses k-nearest-neighbour search
over previously sorted images to suggest a destination folder.

Cache file stores both embeddings and sort history so suggestions
improve across sessions without re-sorting.

Dependencies: transformers, torch  (pip install transformers torch)
"""

import threading
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

CACHE_FILENAME = ".clip_cache.pkl"
MODEL_NAME = "openai/clip-vit-base-patch32"
K_NEIGHBORS = 5
SEED_PER_FOLDER = 10
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


class ClipSuggester:
    def __init__(self, sort_dir: Path, on_ready=None):
        """
        Parameters
        ----------
        sort_dir  : working directory (cache file is written here)
        on_ready  : zero-arg callback fired on the *background* thread when
                    the model finishes loading — use root.after(0, cb) to
                    forward to the main thread.
        """
        self.sort_dir = sort_dir
        self.on_ready = on_ready
        self.ready = False

        self._model = None
        self._processor = None
        # filename -> unit-normalised embedding (ndarray)
        self._emb_cache: dict[str, np.ndarray] = {}
        # (filename, folder_name) pairs saved across sessions
        self._sort_history: list[tuple[str, str]] = []
        # (embedding, folder_name) pairs available for kNN search
        self._sorted: list[tuple[np.ndarray, str]] = []
        self._lock = threading.Lock()

        self._load_cache()
        threading.Thread(target=self._load_model, daemon=True).start()

    # ------------------------------------------------------------------ cache

    def _load_cache(self):
        path = self.sort_dir / CACHE_FILENAME
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # New format: {"embeddings": {...}, "history": [...]}
            # Old format: plain dict of embeddings
            if isinstance(data, dict) and "embeddings" in data:
                self._emb_cache = data["embeddings"]
                self._sort_history = data.get("history", [])
            else:
                self._emb_cache = data
        except Exception:
            pass

    def _save_cache(self):
        try:
            with open(self.sort_dir / CACHE_FILENAME, "wb") as f:
                pickle.dump({"embeddings": self._emb_cache, "history": self._sort_history}, f)
        except Exception:
            pass

    # ------------------------------------------------------------------ model loading

    def _load_model(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained(MODEL_NAME)
            self._model = CLIPModel.from_pretrained(MODEL_NAME)
            self._model.eval()
        except Exception as exc:
            print(f"[CLIP] model load failed: {exc}")
            return
        self.ready = True
        # Rebuild kNN pool from cached history so suggestions work immediately
        self._rebuild_sorted()
        if self.on_ready:
            self.on_ready()

    def _rebuild_sorted(self):
        """Populate _sorted from cached embeddings + sort history."""
        with self._lock:
            self._sorted = [
                (self._emb_cache[fname], folder)
                for fname, folder in self._sort_history
                if fname in self._emb_cache
            ]

    # ------------------------------------------------------------------ embedding

    def _compute_embedding(self, path: Path) -> "np.ndarray | None":
        """Compute a unit-normalised CLIP image embedding. Call from bg thread."""
        try:
            import torch
            img = Image.open(path).convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt")
            with torch.no_grad():
                feat = self._model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze().cpu().numpy()
        except Exception:
            return None

    def embed(self, path: Path) -> "np.ndarray | None":
        """Return cached embedding, or compute + cache it. Call from bg thread."""
        if not self.ready:
            return None
        key = path.name
        with self._lock:
            if key in self._emb_cache:
                return self._emb_cache[key]
        emb = self._compute_embedding(path)
        if emb is not None:
            with self._lock:
                self._emb_cache[key] = emb
            self._save_cache()
        return emb

    # ------------------------------------------------------------------ public API

    def seed_from_folders(self, folders: list[Path], n: int = SEED_PER_FOLDER):
        """
        Bootstrap the kNN pool from images already inside destination folders.
        Takes up to *n* images per folder. Call from a background thread.
        """
        for folder in folders:
            images = [
                f for f in sorted(folder.iterdir())
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ][:n]
            for path in images:
                emb = self.embed(path)
                if emb is not None:
                    with self._lock:
                        self._sorted.append((emb, folder.name))

    def precompute(self, paths: list[Path], on_progress=None):
        """
        Pre-compute and cache embeddings for *paths* in order.
        on_progress(done, total) is called after each image (from this thread).
        Call from a background thread.
        """
        total = len(paths)
        for i, path in enumerate(paths):
            if path.exists():
                self.embed(path)
            if on_progress:
                on_progress(i + 1, total)

    def record(self, path: Path, folder: str):
        """Record that *path* was sorted into *folder*. Call from bg thread."""
        emb = self.embed(path)
        if emb is not None:
            with self._lock:
                self._sorted.append((emb, folder))
                self._sort_history.append((path.name, folder))
            self._save_cache()

    def suggest(self, path: Path) -> "tuple[str, float] | None":
        """
        Return (folder_name, confidence) for the given image, or None if
        there are no sorted examples yet. confidence is in [0, 1].
        Call from a background thread.
        """
        if not self.ready:
            return None
        with self._lock:
            if not self._sorted:
                return None
            sorted_copy = list(self._sorted)

        emb = self.embed(path)
        if emb is None:
            return None

        # Cosine similarities (embeddings are already unit-normalised)
        sims = [(float(np.dot(emb, e)), folder) for e, folder in sorted_copy]
        sims.sort(reverse=True)

        k = min(K_NEIGHBORS, len(sims))
        neighbors = sims[:k]

        # Weighted vote: folder with highest total similarity wins
        scores: dict[str, float] = {}
        for sim, folder in neighbors:
            scores[folder] = scores.get(folder, 0.0) + sim

        best_folder = max(scores, key=scores.__getitem__)
        # Normalise: best possible = k × 1.0
        confidence = scores[best_folder] / k
        return best_folder, confidence
