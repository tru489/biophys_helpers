"""
analysis_wizard.py

One-stop-shop wizard that walks through the full analysis pipeline as a
sequence of "step" pages with Previous/Next navigation, embedding the
existing per-stage GUIs rather than reimplementing them:

    1. Buoyant mass       — placeholder: run the MATLAB SMR/buoyant-mass
                             analysis separately first.
    2. Baseline density   — placeholder: run the imaging/FXM exclusion
                             analysis separately, alongside
                             calculate_baseline_density.py's GUI (embedded)
                             for the fluid baseline density calculation. A
                             valid experiment directory must be set here
                             before Next becomes available.
    3. Gate: Buoyant Mass — gate_experiments_inplace.py embedded in 'bm'
                             mode. Auto-discovers as soon as the directory
                             from step 2 is in place (no Discover click
                             needed). Skipped if the shared directory has
                             volume data but no mass data.
    4. Gate: iFXM Volume  — the same script embedded in 'ifxm' mode, sharing
                             the directory from step 2/3 (one StringVar) and
                             auto-discovering the same way. Skipped if the
                             shared directory has mass data but no volume
                             data.
    5. Compile experiment — compile_experiment.py embedded; its directory
                             field is seeded from the shared directory the
                             first time this page is built, but (unlike
                             steps 3/4) still requires an explicit "Discover
                             Samples" click. Clicking its Done button closes
                             the wizard.

Steps 2-4 share one experiment-directory field (typing in any one updates
the others live); step 5 seeds from it once. Steps 3 and 4 each require
their embedded gating panel's "Finalize gating" to be clicked before Next
becomes available — re-running Discover on either page (a new directory, or
just starting over) revokes that until Finalize is clicked again.

Each embedded page is built by a `build_embedded_page(parent, ...)` function
that lives in the corresponding script (see that script's own module
docstring), so every script's standalone CLI/GUI entry point keeps working
unmodified. Pages are built lazily on first visit and cached (shown/hidden
rather than destroyed) so in-progress work is preserved when navigating back
and forth.

Usage:
    python analysis_wizard.py
"""
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import calculate_baseline_density
import compile_experiment
import gate_experiments_inplace


# ---------------------------------------------------------------------------
# Static text pages
# ---------------------------------------------------------------------------

def _text_page(parent: tk.Widget, text: str) -> ttk.Frame:
    page = ttk.Frame(parent)
    ttk.Label(page, text=text, wraplength=560, justify='left',
             font=('TkDefaultFont', 11)).pack(
        anchor='nw', padx=24, pady=24)
    return page


_BUOYANT_MASS_TEXT = (
    "Step 1 — Buoyant mass analysis\n\n"
    "The MATLAB-based buoyant mass (SMR) analysis needs to be run "
    "separately, outside this GUI, before continuing.\n\n"
    "Once each sample's *_mass_results folder has been produced, move on "
    "to the next step."
)

_BASELINE_DENSITY_TEXT = (
    "Step 2 — Baseline density\n\n"
    "The imaging / fluorescence-exclusion (FXM) analysis also needs to be "
    "run separately, outside this GUI.\n\n"
    "The fluid baseline density calculation below can be run alongside "
    "that separate analysis, using each sample's buoyant mass data."
)


# ---------------------------------------------------------------------------
# Mass/volume detection, for skipping the BM or iFXM gating step when the
# shared directory clearly has only one of the two.
# ---------------------------------------------------------------------------

class _DataAvailability:
    """
    Caches, per directory string, whether mass and volume data were found —
    so Next/Previous can decide which of the BM/iFXM gating steps to skip
    without re-running full discovery (CSV/hdf5 reads) on every keystroke in
    the directory field. Call `refresh()` whenever the shared directory
    might have changed since the last check; pass `force=True` to bypass the
    cache and re-run discovery even if the directory string is unchanged —
    used right when Next/Previous is clicked to actually cross into the BM
    or iFXM gating step, since the separate mass/volume analyses may have
    been run (producing new files) after the directory was first set here.
    """

    def __init__(self):
        self._dir = None
        self._has_mass = True
        self._has_volume = True

    def refresh(self, superdir_text: str, force: bool = False):
        superdir_text = (superdir_text or '').strip()
        if not force and superdir_text == self._dir:
            return
        self._dir = superdir_text
        if not superdir_text or not Path(superdir_text).is_dir():
            # No (valid) directory yet — never skip; let the user visit
            # either gating step to set one.
            self._has_mass = True
            self._has_volume = True
            return
        superdir = Path(superdir_text)
        try:
            self._has_mass = bool(gate_experiments_inplace._discover_bm_tables(superdir))
        except Exception:
            self._has_mass = True
        try:
            self._has_volume = bool(gate_experiments_inplace._discover_ifxm(superdir))
        except Exception:
            self._has_volume = True

    @property
    def skip_bm_step(self) -> bool:
        """Volume data exists but no mass data — nothing to gate on step 3."""
        return self._has_volume and not self._has_mass

    @property
    def skip_ifxm_step(self) -> bool:
        """Mass data exists but no volume data — nothing to gate on step 4."""
        return self._has_mass and not self._has_volume


# ---------------------------------------------------------------------------
# Wizard shell
# ---------------------------------------------------------------------------

class Wizard:
    """
    Minimal step-wizard shell: a title label, a content area holding one
    page at a time, and a Previous/Next bar.

    Each step is a dict:
        title:            shown in the header
        builder:          builder(parent) -> ttk.Frame, called once per page
                          on first visit; the frame is cached and toggled
                          with pack/pack_forget on later visits so page
                          state survives navigation.
        requires_finalize: if True, Next stays disabled on this step until
                          mark_complete(index) is called (and is disabled
                          again by mark_incomplete(index)).
        skip_check:       optional callable(force=False) -> bool; Next/Previous
                          skip over this step whenever it returns True. Called
                          with force=False (cache allowed) for button-state
                          bookkeeping, and force=True (cache bypassed, re-run
                          discovery) at the moment Next/Previous actually
                          crosses over this step, so a skip decided from a
                          stale cache never happens.
        next_check:       optional callable() -> bool; Next stays disabled
                          on this step whenever it returns False (checked
                          live — call refresh_nav() when something this
                          depends on changes, e.g. a directory field).
    """

    def __init__(self, root: tk.Tk, steps: list):
        self._root = root
        self._steps = steps
        self._pages: dict[int, ttk.Frame] = {}
        self._completed: dict[int, bool] = {}
        self._index = 0

        self._title_var = tk.StringVar()
        ttk.Label(root, textvariable=self._title_var,
                 font=('TkDefaultFont', 13, 'bold')).pack(
            anchor='w', padx=12, pady=(12, 4))

        self._content = ttk.Frame(root)
        self._content.pack(fill=tk.BOTH, expand=True, padx=12)

        nav = ttk.Frame(root)
        nav.pack(fill=tk.X, padx=12, pady=12)
        self._prev_btn = ttk.Button(nav, text='← Previous', command=self._go_prev)
        self._prev_btn.pack(side=tk.LEFT)
        self._hint_var = tk.StringVar()
        ttk.Label(nav, textvariable=self._hint_var, foreground='#a00').pack(
            side=tk.LEFT, expand=True)
        self._next_btn = ttk.Button(nav, text='Next →', command=self._go_next)
        self._next_btn.pack(side=tk.RIGHT)

        self._show(self._first_visible_index(0, +1))

    # -- step completion (for requires_finalize steps) ----------------------

    def mark_complete(self, index: int):
        self._completed[index] = True
        if self._index == index:
            self._update_nav()

    def mark_incomplete(self, index: int):
        self._completed[index] = False
        if self._index == index:
            self._update_nav()

    def refresh_nav(self):
        """Re-evaluate the current step's `next_check` (call after something
        it depends on changes, e.g. a shared directory field)."""
        self._update_nav()

    # -- navigation -----------------------------------------------------

    def _is_skipped(self, index: int, force: bool = False) -> bool:
        check = self._steps[index].get('skip_check')
        return bool(check and check(force=force))

    def _first_visible_index(self, start: int, direction: int) -> int:
        """The first in-range, non-skipped index reachable from `start`
        stepping by `direction` (inclusive of `start`)."""
        idx = start
        while 0 <= idx < len(self._steps) and self._is_skipped(idx):
            idx += direction
        if 0 <= idx < len(self._steps):
            return idx
        return start   # nothing else visible in that direction — stay put

    def _show(self, index: int):
        if index not in self._pages:
            self._pages[index] = self._steps[index]['builder'](self._content)
        for i, page in self._pages.items():
            if i == index:
                page.pack(fill=tk.BOTH, expand=True)
            else:
                page.pack_forget()

        self._index = index
        n = len(self._steps)
        self._title_var.set(f"Step {index + 1} of {n} — {self._steps[index]['title']}")
        self._update_nav()

    def _step_ready(self, index: int) -> bool:
        """Whether this step's own requirements (finalize / next_check) are
        satisfied, i.e. whether Next may leave it."""
        step = self._steps[index]
        if step.get('requires_finalize', False) and not self._completed.get(index, False):
            return False
        check = step.get('next_check')
        if check is not None and not check():
            return False
        return True

    def _update_nav(self):
        index = self._index
        n = len(self._steps)
        has_prev = any(not self._is_skipped(i) for i in range(index))
        has_next = any(not self._is_skipped(i) for i in range(index + 1, n))
        self._prev_btn.config(state=tk.NORMAL if has_prev else tk.DISABLED)

        ready = self._step_ready(index)
        self._next_btn.config(state=tk.NORMAL if has_next and ready else tk.DISABLED)

        step = self._steps[index]
        if has_next and not ready:
            if step.get('requires_finalize', False) and not self._completed.get(index, False):
                hint = 'Click "Finalize gating" below to continue.'
            else:
                hint = step.get('next_hint', 'Complete this step to continue.')
        else:
            hint = ''
        self._hint_var.set(hint)

    def _go_prev(self):
        idx = self._index - 1
        while idx >= 0 and self._is_skipped(idx, force=True):
            idx -= 1
        if idx >= 0:
            self._show(idx)

    def _go_next(self):
        if not self._step_ready(self._index):
            return   # guard directly, don't rely solely on the button's disabled state
        idx = self._index + 1
        while idx < len(self._steps) and self._is_skipped(idx, force=True):
            idx += 1
        if idx < len(self._steps):
            self._show(idx)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    root.title('Biophysics Analysis Wizard')
    root.minsize(1000, 700)

    # Shared across steps 2 (baseline density), 3 (gate BM) and 4 (gate
    # iFXM): each page's directory field is bound to this same StringVar, so
    # typing in any one updates the others live. The compile page (step 5)
    # only reads its current value once, as an initial seed, since
    # compilation may reasonably target a different directory afterward.
    experiment_superdir = tk.StringVar(value='')
    availability = _DataAvailability()

    def build_baseline_density(parent):
        page = ttk.Frame(parent)
        ttk.Label(page, text=_BASELINE_DENSITY_TEXT, wraplength=760,
                 justify='left').pack(anchor='nw', padx=8, pady=(8, 12))
        panel = calculate_baseline_density.build_embedded_page(
            page, superdir_var=experiment_superdir)
        panel.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        return page

    def build_gate_bm(parent):
        return gate_experiments_inplace.build_embedded_page(
            parent, 'bm', superdir_var=experiment_superdir,
            on_finalized=lambda: wiz.mark_complete(2),
            on_discover=lambda: wiz.mark_incomplete(2))

    def build_gate_ifxm(parent):
        return gate_experiments_inplace.build_embedded_page(
            parent, 'ifxm', superdir_var=experiment_superdir,
            on_finalized=lambda: wiz.mark_complete(3),
            on_discover=lambda: wiz.mark_incomplete(3))

    def build_compile(parent):
        return compile_experiment.build_embedded_page(
            parent, initial_superdir=experiment_superdir.get(),
            on_done=root.destroy)

    def _skip_bm(force=False):
        availability.refresh(experiment_superdir.get(), force=force)
        return availability.skip_bm_step

    def _skip_ifxm(force=False):
        availability.refresh(experiment_superdir.get(), force=force)
        return availability.skip_ifxm_step

    def _has_superdir():
        text = experiment_superdir.get().strip()
        return bool(text) and Path(text).is_dir()

    steps = [
        {'title': 'Buoyant Mass',
         'builder': lambda parent: _text_page(parent, _BUOYANT_MASS_TEXT)},
        {'title': 'Baseline Density', 'builder': build_baseline_density,
         'next_check': _has_superdir,
         'next_hint': 'Choose an experiment directory above to continue.'},
        {'title': 'Gate: Buoyant Mass', 'builder': build_gate_bm,
         'requires_finalize': True, 'skip_check': _skip_bm},
        {'title': 'Gate: iFXM Volume', 'builder': build_gate_ifxm,
         'requires_finalize': True, 'skip_check': _skip_ifxm},
        {'title': 'Compile Experiment', 'builder': build_compile},
    ]

    wiz = Wizard(root, steps)
    # Re-check step 2's "a directory is required" gate live as the shared
    # field changes (typing, Browse, or being set from another page).
    experiment_superdir.trace_add('write', lambda *_a: wiz.refresh_nav())
    root.mainloop()


if __name__ == '__main__':
    main()
