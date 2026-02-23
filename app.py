import sys
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
from PIL import Image, ImageTk

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
MAX_FOLDERS = 15

try:
    from clip_suggester import ClipSuggester
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


def get_sort_dir() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_dir():
            return p
    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askdirectory(title="Select folder to sort")
    root.destroy()
    if not chosen:
        sys.exit(0)
    return Path(chosen)


class ImageSorter:
    def __init__(self, root: tk.Tk, sort_dir: Path):
        self.root = root
        self.sort_dir = sort_dir
        self.image_list: list[str] = []
        self.sorted_count = 0
        self.total_count = 0
        self.undo_stack: list[dict] = []
        self._photo = None
        self._current_img: Image.Image | None = None
        self.folders: list[Path] = []
        self._folder_buttons: list[tk.Button] = []
        self._suggested_folder: str | None = None
        # incremented each time we navigate to a new image; lets us discard
        # stale suggestion results that arrive after the image has changed
        self._suggestion_id = 0
        self._suggester = None

        self._build_ui()
        self._init_clip()
        self._load_image_list()
        self._refresh_folders()
        self._refresh_ui()
        self._bind_keys()

    # ------------------------------------------------------------------ CLIP

    def _init_clip(self):
        if not CLIP_AVAILABLE:
            self.lbl_clip.config(text="CLIP not installed — pip install transformers torch")
            return
        self.lbl_clip.config(text="CLIP: loading model…")
        self._suggester = ClipSuggester(
            self.sort_dir,
            on_ready=lambda: self.root.after(0, self._on_clip_ready),
        )

    LOOKAHEAD = 10  # number of upcoming images to pre-embed

    def _on_clip_ready(self):
        threading.Thread(target=self._seed_thread, daemon=True).start()

    def _seed_thread(self):
        """Seed kNN pool from existing folder contents, then fire first suggestion."""
        folders = self._get_folders()
        self.root.after(0, lambda: self.lbl_clip.config(text="CLIP: seeding from folders…"))
        self._suggester.seed_from_folders(folders)
        self.root.after(0, lambda: self.lbl_clip.config(text="CLIP: ready"))
        self.root.after(0, self._trigger_suggestion)
        self.root.after(0, self._precompute_lookahead)

    def _precompute_lookahead(self):
        """Pre-embed the next LOOKAHEAD images in the queue (background, silent)."""
        if self._suggester is None or not self._suggester.ready:
            return
        paths = [
            self.sort_dir / f
            for f in self.image_list[1 : self.LOOKAHEAD + 1]
        ]
        if paths:
            threading.Thread(
                target=self._suggester.precompute, args=(paths,), daemon=True
            ).start()

    def _trigger_suggestion(self):
        """Start a background thread to compute a suggestion for the current image."""
        if self._suggester is None or not self._suggester.ready:
            return
        if not self.image_list:
            return
        self._suggestion_id += 1
        req_id = self._suggestion_id
        path = self.sort_dir / self.image_list[0]
        threading.Thread(
            target=self._compute_suggestion,
            args=(path, req_id),
            daemon=True,
        ).start()

    def _compute_suggestion(self, path: Path, req_id: int):
        result = self._suggester.suggest(path)
        self.root.after(0, lambda: self._apply_suggestion(result, req_id))

    def _apply_suggestion(self, result, req_id: int):
        if req_id != self._suggestion_id:
            return  # stale — user already moved on
        if result is None:
            self._suggested_folder = None
            self.lbl_suggestion.config(text="")
        else:
            folder_name, confidence = result
            self._suggested_folder = folder_name
            pct = int(confidence * 100)
            self.lbl_suggestion.config(
                text=f"Suggested: {folder_name}   ({pct}% match)   —   Enter to accept"
            )
        self._highlight_suggested_button()

    def _highlight_suggested_button(self):
        for btn, folder in zip(self._folder_buttons, self.folders):
            if folder.name == self._suggested_folder:
                btn.config(bg="#4a7c4e", fg="white", relief="sunken")
            else:
                btn.config(bg=self._default_btn_bg, fg="black", relief="raised")

    def _accept_suggestion(self):
        if self._suggested_folder is None:
            return
        target = next((f for f in self.folders if f.name == self._suggested_folder), None)
        if target:
            self._move_to(target)

    # ------------------------------------------------------------------ UI build

    def _build_ui(self):
        self.root.title("Image Sorter")
        self.root.minsize(600, 400)
        self.root.columnconfigure(0, weight=1)

        # --- Top bar ---
        top = tk.Frame(self.root)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        top.columnconfigure(0, weight=1)

        self.lbl_filename = tk.Label(top, text="", anchor="w", font=("TkDefaultFont", 10))
        self.lbl_filename.grid(row=0, column=0, sticky="w")

        btn_frame = tk.Frame(top)
        btn_frame.grid(row=0, column=1, sticky="e")

        self.btn_undo = tk.Button(btn_frame, text="Undo  Ctrl+Z", command=self._undo, state="disabled")
        self.btn_undo.pack(side="left", padx=(0, 4))

        self.btn_skip = tk.Button(btn_frame, text="Skip  S →", command=self._skip)
        self.btn_skip.pack(side="left")

        # --- Progress bar row ---
        prog_frame = tk.Frame(self.root)
        prog_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 0))
        prog_frame.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")

        self.lbl_progress = tk.Label(prog_frame, text="0 / 0", width=12, anchor="e")
        self.lbl_progress.grid(row=0, column=1, sticky="e", padx=(6, 0))

        # --- Suggestion row ---
        suggest_frame = tk.Frame(self.root)
        suggest_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 0))
        suggest_frame.columnconfigure(0, weight=1)

        self.lbl_suggestion = tk.Label(
            suggest_frame, text="", anchor="w", fg="#4a7c4e",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.lbl_suggestion.grid(row=0, column=0, sticky="w")

        self.lbl_clip = tk.Label(
            suggest_frame, text="", anchor="e", fg="gray",
            font=("TkDefaultFont", 9),
        )
        self.lbl_clip.grid(row=0, column=1, sticky="e")

        # --- Canvas ---
        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=3, column=0, sticky="nsew", padx=8, pady=8)
        self.root.rowconfigure(3, weight=1)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # --- Done label (hidden until needed) ---
        self.lbl_done = tk.Label(
            self.root, text="",
            font=("TkDefaultFont", 22),
            fg="#4caf50", bg="#1e1e1e",
        )

        # --- Folder button area ---
        self.folder_frame = tk.Frame(self.root)
        self.folder_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))

        # Capture the platform default button background for un-highlighting
        _tmp = tk.Button(self.root)
        self._default_btn_bg = _tmp.cget("bg")
        _tmp.destroy()

    def _bind_keys(self):
        self.root.bind("<KeyPress>", self._on_key)
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Return>", lambda e: self._accept_suggestion())

    # ------------------------------------------------------------------ Data

    def _load_image_list(self):
        files = [
            f.name
            for f in sorted(self.sort_dir.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        self.image_list = files
        self.total_count = len(files) + self.sorted_count

    def _get_folders(self) -> list[Path]:
        return sorted(
            [d for d in self.sort_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name.lower(),
        )[:MAX_FOLDERS]

    def _refresh_folders(self):
        for widget in self.folder_frame.winfo_children():
            widget.destroy()
        self._folder_buttons.clear()

        self.folders = self._get_folders()
        for i, folder in enumerate(self.folders):
            label = f"[{i + 1}]  {folder.name}" if i < 9 else folder.name
            is_suggested = folder.name == self._suggested_folder
            btn = tk.Button(
                self.folder_frame,
                text=label,
                command=lambda f=folder: self._move_to(f),
                relief="sunken" if is_suggested else "raised",
                bg="#4a7c4e" if is_suggested else self._default_btn_bg,
                fg="white" if is_suggested else "black",
                padx=8,
            )
            btn.pack(side="left", padx=3, pady=3)
            self._folder_buttons.append(btn)

    # ------------------------------------------------------------------ UI refresh

    def _refresh_ui(self):
        self.total_count = len(self.image_list) + self.sorted_count
        title = f"Image Sorter — {self.sorted_count} sorted / {self.total_count} total"
        self.root.title(title)

        if not self.image_list:
            self._show_done()
            return

        self._hide_done()
        self.lbl_filename.config(text=self.image_list[0])

        self.progress["maximum"] = max(self.total_count, 1)
        self.progress["value"] = self.sorted_count
        self.lbl_progress.config(text=f"{self.sorted_count} / {self.total_count}")
        self.btn_undo.config(state="normal" if self.undo_stack else "disabled")

        # Clear stale suggestion before loading the new image
        self._suggested_folder = None
        self.lbl_suggestion.config(text="")

        self._load_current_image()
        self._refresh_folders()
        self._trigger_suggestion()

    def _show_done(self):
        self.canvas.grid_remove()
        self.lbl_filename.config(text="")
        self.lbl_suggestion.config(text="")
        self.btn_skip.config(state="disabled")
        self.btn_undo.config(state="normal" if self.undo_stack else "disabled")
        n = self.sorted_count
        self.lbl_done.config(text=f"✓  All done!  {n} image{'s' if n != 1 else ''} sorted.")
        self.lbl_done.grid(row=3, column=0, sticky="nsew", padx=8, pady=8)

    def _hide_done(self):
        self.lbl_done.grid_remove()
        self.canvas.grid()
        self.btn_skip.config(state="normal")

    # ------------------------------------------------------------------ Image rendering

    def _load_current_image(self):
        if not self.image_list:
            self._current_img = None
            return
        path = self.sort_dir / self.image_list[0]
        try:
            img = Image.open(path)
            img.load()
            self._current_img = img
        except Exception:
            self._current_img = None
        self._render_image()

    def _render_image(self):
        self.canvas.delete("all")
        if self._current_img is None:
            cx = self.canvas.winfo_width() // 2 or 300
            cy = self.canvas.winfo_height() // 2 or 200
            self.canvas.create_text(cx, cy, text="(could not load image)", fill="gray")
            return

        cw = self.canvas.winfo_width() or 600
        ch = self.canvas.winfo_height() or 400
        iw, ih = self._current_img.size
        ratio = min(cw / iw, ch / ih)
        new_w = max(1, int(iw * ratio))
        new_h = max(1, int(ih * ratio))

        resized = self._current_img.resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self._photo)

    def _on_canvas_resize(self, event):
        self._render_image()

    # ------------------------------------------------------------------ Actions

    def _move_to(self, folder: Path):
        if not self.image_list:
            return

        filename = self.image_list[0]
        src = self.sort_dir / filename
        if not src.exists():
            self.image_list.pop(0)
            self._refresh_ui()
            return

        # Resolve filename collision
        dst = folder / filename
        if dst.exists():
            counter = 1
            while dst.exists():
                dst = folder / f"{src.stem}_{counter}{src.suffix}"
                counter += 1

        shutil.move(str(src), str(dst))

        # Record sort for CLIP learning (background thread)
        if self._suggester:
            threading.Thread(
                target=self._suggester.record,
                args=(dst, folder.name),
                daemon=True,
            ).start()

        self.undo_stack.append({
            "current_path": dst,
            "original_path": src,
            "filename": filename,
        })
        self.image_list.pop(0)
        self.sorted_count += 1
        self._refresh_ui()
        self._precompute_lookahead()

    def _skip(self):
        if len(self.image_list) <= 1:
            return
        self.image_list.append(self.image_list.pop(0))
        self._refresh_ui()
        self._precompute_lookahead()

    def _undo(self):
        if not self.undo_stack:
            return
        entry = self.undo_stack.pop()
        src = Path(entry["current_path"])
        dst = Path(entry["original_path"])
        if src.exists():
            shutil.move(str(src), str(dst))
        self.image_list.insert(0, entry["filename"])
        self.sorted_count -= 1
        self._refresh_ui()

    # ------------------------------------------------------------------ Keyboard

    def _on_key(self, event: tk.Event):
        key = event.keysym.lower()
        if key in ("s", "right"):
            self._skip()
        elif key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            idx = int(key) - 1
            if idx < len(self.folders):
                self._move_to(self.folders[idx])


def main():
    sort_dir = get_sort_dir()
    root = tk.Tk()
    root.geometry("900x700")
    ImageSorter(root, sort_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
