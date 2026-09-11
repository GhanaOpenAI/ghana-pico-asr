"""Language registry.

The acoustic model is language-agnostic — it maps audio frames to unit classes.
Everything language-specific lives in a :class:`Language`, so supporting a new
language means adding a module here and registering it.

Checkpoints record their language code, so inference reconstructs the right
inventory without the caller having to know it.
"""

from __future__ import annotations

from .base import Language, build_tokens_starred, flatten
from .twi import TWI

#: Every supported language, by code.
LANGUAGES: dict[str, Language] = {TWI.code: TWI}

DEFAULT_LANGUAGE = TWI.code

__all__ = [
    "Language",
    "LANGUAGES",
    "DEFAULT_LANGUAGE",
    "get_language",
    "flatten",
    "build_tokens_starred",
]


def get_language(code: str | None = None) -> Language:
    """Look up a language by code, with a clear error listing the options."""
    code = (code or DEFAULT_LANGUAGE).lower()
    # Accept a few aliases people will reasonably type.
    aliases = {"tw": "twi", "ak": "twi", "akan": "twi", "asante": "twi", "twi_asante": "twi"}
    code = aliases.get(code, code)
    if code not in LANGUAGES:
        raise KeyError(
            f"unsupported language {code!r}; available: {sorted(LANGUAGES)}. "
            "Add one in ghana_pico_asr/languages/ and register it in LANGUAGES."
        )
    return LANGUAGES[code]
