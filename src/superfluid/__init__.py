"""Superfluid: cellular-sheaf diffusion as an attention replacement."""

from .autodiff import Tensor, check_grads, index_add, numerical_grad
from .diffusion import (band_filter_bank, chebyshev_basis, expm_multiply,
                        gershgorin_bound, jam_time, spectral_gap)
from .sheaf import (balance_residual, laplacian_from_edges,
                    laplacian_tensor, total_variation)

__version__ = "0.1.0"