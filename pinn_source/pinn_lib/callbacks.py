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
        from matplotlib.ticker import LogLocator, NullFormatter

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 5),
                                                 sharey=True)
        ax_train.set_title('Training')
        ax_test.set_title('Test')

        # Separate train vs test losses
        for name, entry in pb.history.get('losses', {}).items():
            vals = entry.get('log', [])
            iters = entry.get('iter', [])
            if not vals:
                continue
            x = iters if len(iters) == len(vals) else list(range(len(vals)))
            is_test = name.startswith('test/')
            label = name.removeprefix('test/') if is_test else name
            ax = ax_test if is_test else ax_train
            ax.plot(x, vals, label=label, linewidth=0.9)

        # Optimizer transition lines
        transitions = pb.history.get('transitions', [])
        for ax in (ax_train, ax_test):
            for t in transitions:
                it = t['iter']
                if it > 0:
                    ax.axvline(it, color='grey', linewidth=0.6,
                               linestyle=':', alpha=0.8)
                    ax.text(it, 1.02, t['method'], transform=ax.get_xaxis_transform(),
                            ha='left', va='bottom', fontsize=6, color='grey')

        # Log-log scales and formatting
        for ax in (ax_train, ax_test):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Iteration')
            ax.legend(loc='upper right', fontsize=7, ncol=1)
            ax.grid(True, which='major', linewidth=0.4, alpha=0.5)
            ax.grid(True, which='minor', linewidth=0.2, alpha=0.3)
        ax_train.set_ylabel('Loss')

        fig.suptitle('Loss History', fontsize=12)
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
