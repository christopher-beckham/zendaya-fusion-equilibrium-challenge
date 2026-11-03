"""gsolve: a small free-boundary Grad-Shafranov solver (batched, torch on CPU) for MAST and DIII-D."""
from .correct import RICH, SIMPLE, CorrConfig, correct
from .diagnostics import analyse, analyse_all
from .forward import FwdConfig, Solution, balance_height, forward_solve, start_current
from .geometry import Machine, load_machine
from .greens import cell_matrix, coil_flux, coil_matrix, greens

__all__ = ["CorrConfig", "FwdConfig", "Machine", "RICH", "SIMPLE", "Solution", "analyse", "analyse_all", "balance_height",
           "cell_matrix", "coil_flux", "coil_matrix", "correct", "forward_solve", "greens", "load_machine", "start_current"]
__version__ = "0.1.0"
