import torch
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz, csr_matrix
from typing import Optional

__all__ = [
    "load_math_flag_csr",
    "build_math_token_mask",
]

_MATH_CSR = None  # type: Optional[csr_matrix]
_MATH_ROW_BOOL = None  # type: Optional[np.ndarray]  # numpy bool array len=vocab

_DEFAULT_MATH_CSR = (
    "/opt/aps/workdir/jiechen/train/token_math_mask/token_math_flag_csr.npz"
)


def load_math_flag_csr(path: str | Path = _DEFAULT_MATH_CSR):
    """Load CSR matrix and precompute fast row-bool lookup array.

    Returns the CSR matrix (on CPU). A side-effect is filling `_MATH_ROW_BOOL` for
    O(1) masking.
    """
    global _MATH_CSR, _MATH_ROW_BOOL
    if _MATH_CSR is None:
        _MATH_CSR = load_npz(Path(path)).tocsr()
        # precompute boolean: True if row has non-zero (i.e., math token)
        _MATH_ROW_BOOL = (_MATH_CSR.indptr[1:] - _MATH_CSR.indptr[:-1]) > 0
        print(
            f"[load_matrix] math_flag_csr loaded: shape={_MATH_CSR.shape}, nnz={_MATH_CSR.nnz}"
        )
    else:
        print(
            f"[load_matrix] math_flag_csr already loaded: shape={_MATH_CSR.shape}, nnz={_MATH_CSR.nnz}"
        )
    return _MATH_CSR


def build_math_token_mask(responses: torch.LongTensor) -> torch.Tensor:
    """Return (bs,L) bool mask marking math tokens.

    Operates on CPU—only the final mask tensor is moved to *responses.device*.
    Out-of-vocabulary token ids are treated as non-math (False).
    """
    # Ensure resources loaded
    load_math_flag_csr()
    assert _MATH_ROW_BOOL is not None

    row_bool = _MATH_ROW_BOOL  # numpy bool
    vocab = row_bool.shape[0]

    ids_flat = responses.detach().cpu().numpy().ravel()
    mask_np = np.zeros_like(ids_flat, dtype=np.bool_)

    in_range = ids_flat < vocab
    if in_range.any():
        mask_np[in_range] = row_bool[ids_flat[in_range]]

    mask = torch.from_numpy(mask_np).view_as(responses)
    return mask.to(responses.device)
