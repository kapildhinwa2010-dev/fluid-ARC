"""Superfluid: cellular-sheaf diffusion as an attention replacement."""

from .autodiff import Tensor, check_grads, index_add, numerical_grad
from .diffusion import (band_filter_bank, chebyshev_basis, expm_multiply,
                        gershgorin_bound, jam_time, spectral_gap)
from .sheaf import (balance_residual, laplacian_from_edges,
                    laplacian_tensor, total_variation)
from .topology import (UnionFind, betti_numbers, component_ids, edge_mask,
                       harmonic_projection, kruskal_barcode,
                       persistence_descriptor, topk_edges)

__version__ = "0.1.0"