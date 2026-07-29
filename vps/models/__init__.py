from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from vps.models.pose_model_contract import BasePoseModel, PoseModelOutput
from vps.models.pose_model import PoseModel
from vps.models.vpr_model import VPRModel

if TYPE_CHECKING:
    from vps.models.omnivggt_model import OmniVGGTModel
    from vps.models.vggt_omega_model import VGGTOmegaModel
    from vps.models.vggt_model import VGGTModel


_OPTIONAL_MODEL_EXPORTS = {
    "OmniVGGTModel": ("vps.models.omnivggt_model", "OmniVGGTModel"),
    "VGGTOmegaModel": ("vps.models.vggt_omega_model", "VGGTOmegaModel"),
    "VGGTModel": ("vps.models.vggt_model", "VGGTModel"),
}


def __getattr__(name: str):
    """Import optional pose backends only when an external caller asks for one."""
    target = _OPTIONAL_MODEL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value

__all__ = [
    "BasePoseModel",
    "OmniVGGTModel",
    "PoseModelOutput",
    "PoseModel",
    "VGGTOmegaModel",
    "VGGTModel",
    "VPRModel",
]
