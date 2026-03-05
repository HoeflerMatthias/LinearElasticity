"""Training callbacks — drop-in replacement for ``nisaba.utils``."""

import json
import os

import numpy as np


class HistoryPlotCallback:
    """Logs loss history, saves a JSON file and a loss-curve plot.

    Parameters
    ----------
    gui : bool
        Ignored (kept for API compatibility).
    filename : str or None
        Path for the PNG loss-curve plot.
    filename_history : str or None
        Path for the JSON history dump.
    frequency : int
        Print a summary line every *frequency* iterations.
    """

    def __init__(self, gui=False, filename=None, filename_history=None, frequency=10):
        self.gui = gui
        self.filename = filename
        self.filename_history = filename_history
        self.frequency = frequency

    def __call__(self, pb, itr, itr_round):
        pass  # logging handled by the optimisation loop

    # ------------------------------------------------------------------ #
    #  finalize — called once after training
    # ------------------------------------------------------------------ #

    def finalize(self, pb, block=False):
        if self.filename_history:
            self._save_history(pb)
        if self.filename:
            self._plot_history(pb)

    # ------------------------------------------------------------------ #
    #  internals
    # ------------------------------------------------------------------ #

    def _save_history(self, pb):
        os.makedirs(os.path.dirname(self.filename_history), exist_ok=True)
        with open(self.filename_history, 'w') as fh:
            json.dump(_make_serializable(pb.history), fh, indent=2)

    def _plot_history(self, pb):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        for name, entry in pb.history.get('losses', {}).items():
            vals = entry.get('log', [])
            if vals:
                ax.semilogy(vals, label=name, linewidth=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.set_title('Loss History')

        plt.tight_layout()
        plt.savefig(self.filename, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------- #
#  Serialisation helper
# ---------------------------------------------------------------------- #

def _make_serializable(obj):
    """Recursively convert numpy / TF types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'numpy'):
        arr = obj.numpy()
        if np.ndim(arr) == 0:
            return float(arr)
        return arr.tolist()
    return obj
