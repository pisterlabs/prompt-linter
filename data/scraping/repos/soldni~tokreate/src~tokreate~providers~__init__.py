from necessary import necessary

from . import anthropic, toghether, tulu  # noqa: F401
from .base import ProviderMessage, ProviderRegistry, ProviderResult

if necessary("openai>=1.0.0", soft=True):
    from . import openai_v1  # noqa: F401
else:
    from . import openai_v0  # noqa: F401


__all__ = ["ProviderRegistry", "ProviderMessage", "ProviderResult"]
