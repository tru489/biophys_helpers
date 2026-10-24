"""Scaffold a new biophys analysis and pull in its compiled raw data.

Lists the subdirs of RAW_DATA_DIR (newest first) to pick from; the chosen
dir's name becomes the analysis name, and it creates
  - <BIOPHYS_CODE_DIR>/<name>/                (code, in the scratch repo)
  - <DATA_ANALYSIS_DIR>/<name>/{data,fig}/    (data + figures)
then looks in the chosen raw dir for
  - *_compiled        -> its contents copied into <name>/data/
  - *_gating_summary  -> copied to <name>/gating/
and shows what it found. Nothing is copied until you click Copy.

Edit the paths below to match this computer's setup.

Embedding: `build_embedded_page(parent, raw_dir_var=..., on_close=...)`
builds the same panel inside another app (analysis_wizard.py uses it as its
final step). There the raw-data-dir list is replaced by the host's shared
directory variable plus a "Create" button; everything from Create onward
(discovery, Copy, progress) is identical to the standalone GUI.

Usage:
    python new_analysis_dir.py [filter]
"""

import argparse
import os
import queue
import re
import shutil
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

BIOPHYS_CODE_DIR = Path(r"C:\thomasu\code\scratch\biophys_data_analysis")
DATA_ANALYSIS_DIR = Path(r"C:\Users\thomasu\MIT Dropbox\Thomas Usherwood\data_analysis")
RAW_DATA_DIR = Path(r"D:\raw_data")

CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Target:
    label: str
    pattern: str  # glob matched against the immediate subdirs of the raw dir
    dest: str     # relative to <DATA_ANALYSIS_DIR>/<name>; the matched dir's contents land here


TARGETS = (
    Target("compiled", "*_compiled", "data"),
    Target("gating summary", "*_gating_summary", "gating"),
)


def name_error(name):
    """Return why `name` is unusable as an analysis dir name, or None if it's fine."""
    if not name:
        return "name required"
    if name in (".", ".."):
        return "invalid name"
    if re.search(r'[\\/:*?"<>|]', name):
        return 'name cannot contain \\ / : * ? " < > |'
    if name.startswith("-"):
        return 'name cannot start with "-"'
    if name != name.strip() or name.endswith("."):
        return "name cannot start/end with a space or end with a dot"
    return None


def find_candidates(raw_dir, pattern):
    """Subdirs of raw_dir matching pattern, oldest first (names are timestamp-prefixed)."""
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.glob(pattern) if p.is_dir())


def tree_size(path):
    n_files = n_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            n_bytes += os.path.getsize(os.path.join(root, f))
            n_files += 1
    return n_files, n_bytes


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def copy_trees(jobs, msgs, cancel):
    """Copy each (src_dir, dest_dir) tree's contents into dest_dir, posting progress to msgs.

    Runs on a worker thread. Never overwrites: an existing destination file is an error.
    """
    try:
        files = []
        for src, dest in jobs:
            for root, _, names in os.walk(src):
                rel = Path(root).relative_to(src)
                (dest / rel).mkdir(parents=True, exist_ok=True)
                for nm in names:
                    s = Path(root) / nm
                    files.append((s, dest / rel / nm, s.stat().st_size))
        total = sum(size for *_, size in files)

        done = 0
        for s, d, _ in files:
            msgs.put(("file", str(s)))
            with open(s, "rb") as fi, open(d, "xb") as fo:
                while chunk := fi.read(CHUNK):
                    if cancel.is_set():
                        break
                    fo.write(chunk)
                    done += len(chunk)
                    msgs.put(("progress", done, total))
            if cancel.is_set():
                d.unlink(missing_ok=True)
                msgs.put(("cancelled",))
                return
            shutil.copystat(s, d)
        msgs.put(("done", len(files), total))
    except Exception as e:  # surfaced in the GUI rather than dying silently on the thread
        msgs.put(("error", f"{type(e).__name__}: {e}"))


class NewAnalysisPanel(ttk.Frame):
    """The whole GUI as a Frame.

    Standalone (raw_dir_var=None): step 1 is a filterable list of RAW_DATA_DIR's subdirs.
    Embedded (raw_dir_var given): step 1 is just that variable's dir plus a Create button.
    `on_close` is what the Close button (and on_close() once any copy is cancelled) calls.
    """

    def __init__(self, parent, initial_filter="", raw_dir_var=None, on_close=None):
        super().__init__(parent, padding=12)
        self.name = None
        self.data_dir = None
        self.rows = []  # (target, enabled BooleanVar, candidates, Combobox|None, size Label)
        self.size_cache = {}
        self.msgs = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.quit_when_done = False
        self.cur_file = ""
        self.cur_bytes = ""
        self.raw_dir_var = raw_dir_var
        self._quit = on_close or self.winfo_toplevel().destroy

        outer = self
        outer.columnconfigure(0, weight=1)

        # -- step 1: pick a raw data dir; its name is the analysis name --------
        self.pick = ttk.Frame(outer)
        self.pick.grid(row=0, column=0, sticky="nsew")
        if raw_dir_var is None:
            outer.rowconfigure(0, weight=1)
            self._build_list_picker(initial_filter)
        else:
            self._build_var_picker()

        self.created_lbl = ttk.Label(outer, text="", foreground="gray30", justify="left")
        self.created_lbl.grid(row=1, column=0, sticky="w", pady=(6, 0))

        # -- step 2: discovery (filled in after create) ------------------------
        self.disc = ttk.Frame(outer)
        self.disc.grid(row=2, column=0, sticky="nsew", pady=(10, 0))

    def _build_list_picker(self, initial_filter):
        self.pick.columnconfigure(1, weight=1)
        self.pick.rowconfigure(1, weight=1)
        ttk.Label(self.pick, text=f"Raw data dir in {RAW_DATA_DIR}:").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self.listbox = tk.Listbox(self.pick, height=15, width=60, exportselection=False,
                                  activestyle="dotbox")
        self.listbox.grid(row=1, column=0, columnspan=2, sticky="nsew")
        sb = ttk.Scrollbar(self.pick, orient="vertical", command=self.listbox.yview)
        sb.grid(row=1, column=2, sticky="ns")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<Double-Button-1>", lambda e: self.create())
        self.listbox.bind("<Return>", lambda e: self.create())

        ttk.Label(self.pick, text="Filter:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.filter_var = tk.StringVar(value=initial_filter)
        self.filter_var.trace_add("write", lambda *_: self.refresh_list())
        self.filter_entry = ttk.Entry(self.pick, textvariable=self.filter_var)
        self.filter_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(6, 0), pady=(6, 0))
        self.filter_entry.bind("<Return>", lambda e: self.create())
        self.filter_entry.bind("<Down>", lambda e: self.listbox.focus_set())

        btns = ttk.Frame(self.pick)
        btns.grid(row=3, column=0, columnspan=3, sticky="e", pady=(8, 0))
        self.browse_raw_btn = ttk.Button(btns, text="Browse...", command=self.browse_raw)
        self.browse_raw_btn.grid(row=0, column=0, padx=(0, 6))
        self.create_btn = ttk.Button(btns, text="Create", command=self.create)
        self.create_btn.grid(row=0, column=1)

        self.raw_names = (sorted((p.name for p in RAW_DATA_DIR.iterdir() if p.is_dir()),
                                 reverse=True)  # newest first (names are date-prefixed)
                          if RAW_DATA_DIR.is_dir() else [])
        self.refresh_list()
        self.filter_entry.focus_set()
        self.filter_entry.icursor("end")

    def _build_var_picker(self):
        f = self.pick
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Raw data:", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Label(f, textvariable=self.raw_dir_var).grid(row=0, column=1, sticky="w")
        self.preview_lbl = ttk.Label(f, text="", justify="left")
        self.preview_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.create_btn = ttk.Button(f, text="Create",
                                     command=lambda: self.create(self._var_raw_dir()))
        self.create_btn.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.raw_dir_var.trace_add("write", lambda *_: self.refresh_preview())
        self.refresh_preview()

    def _var_raw_dir(self):
        text = self.raw_dir_var.get().strip()
        return Path(text) if text else None

    def refresh_preview(self):
        """Embedded mode: show what Create would make from the shared dir, and gate the button."""
        raw_dir = self._var_raw_dir()
        err = name_error(raw_dir.name) if raw_dir else "choose an experiment directory first"
        if err is None and not raw_dir.is_dir():
            err = "directory not found"
        lines, color = [], "gray30"
        if err:
            lines.append(err)
            color = "firebrick"
        else:
            for d in (BIOPHYS_CODE_DIR / raw_dir.name, DATA_ANALYSIS_DIR / raw_dir.name):
                if d.exists():
                    lines.append(f"{d}  (already exists)")
                    color = "firebrick"
                else:
                    lines.append(f"Will create {d}")
        self.preview_lbl.config(text="\n".join(lines), foreground=color)
        self.create_btn.state(["!disabled"] if color == "gray30" else ["disabled"])

    # ------------------------------------------------------------------ pick --
    def refresh_list(self):
        q = self.filter_var.get().strip().lower()
        self.shown = [n for n in self.raw_names if q in n.lower()]
        self.listbox.delete(0, "end")
        for i, n in enumerate(self.shown):
            done = (BIOPHYS_CODE_DIR / n).exists() or (DATA_ANALYSIS_DIR / n).exists()
            self.listbox.insert("end", f"{n}   (exists)" if done else n)
            if done:
                self.listbox.itemconfig(i, foreground="gray50")
        if not self.raw_names:
            self.listbox.insert("end", f"(no subdirs found in {RAW_DATA_DIR})")
            self.listbox.itemconfig(0, foreground="firebrick")
        elif self.shown:
            self.listbox.selection_set(0)
            self.listbox.see(0)

    def browse_raw(self):
        start = RAW_DATA_DIR if RAW_DATA_DIR.is_dir() else Path.home()
        d = filedialog.askdirectory(parent=self, initialdir=start,
                                    title="Choose the raw data dir")
        if d:
            self.create(Path(d))

    # ---------------------------------------------------------------- create --
    def create(self, raw_dir=None):
        if raw_dir is None:
            sel = self.listbox.curselection()
            if not self.shown or not sel:
                messagebox.showerror("Nothing selected", "Pick a raw data dir.", parent=self)
                return
            raw_dir = RAW_DATA_DIR / self.shown[sel[0]]
        name = raw_dir.name
        err = name_error(name)
        if err:
            messagebox.showerror("Invalid name", err, parent=self)
            return
        code_dir = BIOPHYS_CODE_DIR / name
        data_dir = DATA_ANALYSIS_DIR / name
        for d in (code_dir, data_dir):
            if d.exists():
                messagebox.showerror("Already exists", f"{d} already exists", parent=self)
                return
        try:
            code_dir.mkdir(parents=True)
            (data_dir / "fig").mkdir(parents=True)
            (data_dir / "data").mkdir()
        except OSError as e:
            messagebox.showerror("Could not create directories", str(e), parent=self)
            return

        self.name, self.data_dir = name, data_dir
        self.pick.grid_remove()
        self.created_lbl.config(text="\n".join(
            f"Created {p}" for p in (code_dir, data_dir / "fig", data_dir / "data")))
        self.show_discovery(raw_dir)

    # ------------------------------------------------------------- discovery --
    def show_discovery(self, raw_dir):
        for w in self.disc.winfo_children():
            w.destroy()
        self.rows = []
        f = self.disc
        f.columnconfigure(1, weight=1)

        src_row = ttk.Frame(f)
        src_row.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        src_row.columnconfigure(1, weight=1)
        ttk.Label(src_row, text="Raw data:", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, padx=(0, 6))
        found_raw = raw_dir.is_dir()
        ttk.Label(src_row, text=str(raw_dir) if found_raw else f"{raw_dir}  (not found)",
                  foreground="black" if found_raw else "firebrick").grid(row=0, column=1, sticky="w")
        self.browse_btn = ttk.Button(src_row, text="Choose other...", command=self.browse)
        self.browse_btn.grid(row=0, column=2, padx=(6, 0))

        for i, t in enumerate(TARGETS, start=1):
            cands = find_candidates(raw_dir, t.pattern)
            var = tk.BooleanVar(value=bool(cands))
            chk = ttk.Checkbutton(f, text=t.label, variable=var, command=self.refresh_copy_btn)
            chk.grid(row=i, column=0, sticky="w", padx=(0, 10), pady=2)
            size_lbl = ttk.Label(f, text="", foreground="gray30")
            combo = None
            if cands:
                combo = ttk.Combobox(f, state="readonly", width=42,
                                     values=[c.name for c in cands])
                combo.current(len(cands) - 1)  # newest
                combo.grid(row=i, column=1, sticky="ew", pady=2)
                combo.bind("<<ComboboxSelected>>", lambda e, r=len(self.rows): self.update_size(r))
            else:
                chk.state(["disabled"])
                ttk.Label(f, text=f"not found ({t.pattern})", foreground="firebrick").grid(
                    row=i, column=1, sticky="w", pady=2)
            size_lbl.grid(row=i, column=2, sticky="w", padx=10)
            ttk.Label(f, text=f"\u2192 {self.name}/{t.dest}/").grid(row=i, column=3, sticky="w")
            self.rows.append((t, var, cands, combo, size_lbl))
            if cands:
                self.update_size(len(self.rows) - 1)

        n = len(TARGETS) + 1
        ttk.Label(f, text=f"Destination: {self.data_dir}", foreground="gray30").grid(
            row=n, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.progress = ttk.Progressbar(f, mode="determinate", maximum=1.0)
        self.progress.grid(row=n + 1, column=0, columnspan=4, sticky="ew", pady=(10, 2))
        self.status = ttk.Label(f, text="Review the above, then Copy.", foreground="gray30")
        self.status.grid(row=n + 2, column=0, columnspan=4, sticky="w")

        btns = ttk.Frame(f)
        btns.grid(row=n + 3, column=0, columnspan=4, sticky="e", pady=(10, 0))
        self.copy_btn = ttk.Button(btns, text="Copy", command=self.start_copy)
        self.copy_btn.grid(row=0, column=0, padx=(0, 6))
        self.cancel_btn = ttk.Button(btns, text="Cancel copy", command=self.cancel.set)
        self.close_btn = ttk.Button(btns, text="Close", command=self.on_close)
        self.close_btn.grid(row=0, column=2)
        self.refresh_copy_btn()
        if not any(r[2] for r in self.rows):
            self.status.config(text="Nothing to copy; the empty analysis dirs were created.")

    def browse(self):
        start = RAW_DATA_DIR if RAW_DATA_DIR.is_dir() else Path.home()
        d = filedialog.askdirectory(parent=self, initialdir=start,
                                    title="Choose the raw data dir")
        if d:
            self.show_discovery(Path(d))

    def selected(self, row):
        _, _, cands, combo, _ = self.rows[row]
        return cands[combo.current()]

    def update_size(self, row):
        path = self.selected(row)
        if path not in self.size_cache:
            self.size_cache[path] = tree_size(path)
        n_files, n_bytes = self.size_cache[path]
        extra = f", {len(self.rows[row][2])} matches" if len(self.rows[row][2]) > 1 else ""
        self.rows[row][4].config(text=f"{n_files} files, {fmt_bytes(n_bytes)}{extra}")

    def refresh_copy_btn(self):
        ok = any(var.get() for _, var, *_ in self.rows)
        self.copy_btn.state(["!disabled"] if ok else ["disabled"])

    # ------------------------------------------------------------------ copy --
    def start_copy(self):
        jobs = [(self.selected(i), self.data_dir / t.dest)
                for i, (t, var, *_) in enumerate(self.rows) if var.get()]
        if not jobs:
            return
        for w in self.disc.winfo_children():
            self._set_enabled(w, False)
        self.close_btn.state(["!disabled"])
        self.copy_btn.grid_remove()
        self.cancel_btn.grid(row=0, column=1, padx=(0, 6))
        self.cancel_btn.state(["!disabled"])
        self.status.config(text="Scanning...")
        self.worker = threading.Thread(target=copy_trees, args=(jobs, self.msgs, self.cancel),
                                       daemon=True)
        self.worker.start()
        self.poll()

    def _set_enabled(self, widget, enabled):
        try:
            widget.state(["!disabled"] if enabled else ["disabled"])
        except (AttributeError, tk.TclError):
            pass
        for child in widget.winfo_children():
            self._set_enabled(child, enabled)

    def poll(self):
        finished = None
        while True:
            try:
                msg = self.msgs.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "file":
                self.cur_file = Path(msg[1]).name
            elif msg[0] == "progress":
                done, total = msg[1], msg[2]
                self.progress["value"] = done / total if total else 1.0
                self.cur_bytes = f"{fmt_bytes(done)} / {fmt_bytes(total)}"
            else:
                finished = msg
        if self.cur_file:
            self.status.config(text=f"{self.cur_bytes}   {self.cur_file}")

        if finished is None:
            self.after(100, self.poll)
            return

        self.cancel_btn.grid_remove()
        kind = finished[0]
        if kind == "done":
            self.progress["value"] = 1.0
            self.status.config(text=f"Copied {finished[1]} files ({fmt_bytes(finished[2])}) "
                                    f"into {self.data_dir}", foreground="darkgreen")
        elif kind == "cancelled":
            self.status.config(text="Copy cancelled; already-copied files were left in place.",
                               foreground="firebrick")
        else:
            self.status.config(text=f"Copy failed: {finished[1]}", foreground="firebrick")
            messagebox.showerror("Copy failed", finished[1], parent=self)
        if self.quit_when_done:
            self._quit()

    def on_close(self):
        """Close, first cancelling (and waiting out) any in-progress copy.

        Hosts should route their window's close button here too, so a copy
        isn't killed mid-file.
        """
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Copy in progress", "Cancel the copy and quit?",
                                       parent=self):
                return
            self.quit_when_done = True
            self.cancel.set()
            return
        self._quit()


def build_embedded_page(parent, *, raw_dir_var, on_close=None):
    """Build this tool's panel for embedding in a larger app, e.g. a wizard page.

    `raw_dir_var` (a tk.StringVar holding the raw data dir) stands in for the
    standalone list picker: its dir's name becomes the analysis name, and its
    *_compiled / *_gating_summary subdirs are what's offered for copying.
    `on_close` is called by the Close button (default: destroy the toplevel).
    Returns the NewAnalysisPanel, whose on_close() the host should also bind
    to its window's close button.
    """
    return NewAnalysisPanel(parent, raw_dir_var=raw_dir_var, on_close=on_close)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filter", nargs="?", default="", help="prefill the raw dir list filter")
    args = ap.parse_args()

    root = tk.Tk()
    root.title("New analysis dir")
    panel = NewAnalysisPanel(root, args.filter)
    panel.pack(fill=tk.BOTH, expand=True)
    root.protocol("WM_DELETE_WINDOW", panel.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
