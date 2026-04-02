"""AutoDeploy's library of transforms.

This file ensures that all publicly listed files/transforms in the library folder are auto-imported
and the corresponding transforms are registered.
"""

import importlib
import pkgutil

from ...utils.logger import ad_logger

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if module_name.startswith("_"):
        continue
    __all__.append(module_name)
    try:
        importlib.import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        ad_logger.warning(
            "Skipping optional AutoDeploy transform module %s due to missing dependency: %s",
            module_name,
            exc,
        )
