"""pinn_lib — lightweight drop-in replacement for the nisaba PINN framework.

Usage::

    import pinn_source.pinn_lib as ns

    ns.config.get_dtype()
    ns.Loss(...)
    ns.LossMeanSquares(...)
    ns.DataSet(...)
    ns.DataCollection(...)
    ns.OptimizationProblem(...)
    ns.minimize(...)
    ns.GradientTape
    ns.utils.HistoryPlotCallback(...)
"""

from pinn_source.pinn_lib import config
from pinn_source.pinn_lib import loss
from pinn_source.pinn_lib import callbacks as utils

from pinn_source.pinn_lib.loss import Loss, LossMeanSquares
from pinn_source.pinn_lib.data import DataSet, DataCollection
from pinn_source.pinn_lib.optimization import OptimizationProblem, minimize
from pinn_source.pinn_lib.gradient_tape import GradientTape
