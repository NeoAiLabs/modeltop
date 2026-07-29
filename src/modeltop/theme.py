"""Shared Catppuccin palettes for Textual and Rich renderables."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

CatppuccinTheme = Literal[
    "catppuccin-latte",
    "catppuccin-frappe",
    "catppuccin-macchiato",
    "catppuccin-mocha",
]

DEFAULT_CATPPUCCIN_THEME: Final[CatppuccinTheme] = "catppuccin-mocha"


@dataclass(frozen=True)
class CatppuccinPalette:
    """Semantic colors shared by Rich renderables and custom TCSS roles."""

    crust: str
    base: str
    text: str
    muted: str
    primary: str
    secondary: str
    accent: str
    warning: str
    error: str
    success: str


CATPPUCCIN_PALETTES: Final[Mapping[CatppuccinTheme, CatppuccinPalette]] = {
    "catppuccin-latte": CatppuccinPalette(
        "#dce0e8",
        "#eff1f5",
        "#4c4f69",
        "#6c6f85",
        "#8839ef",
        "#dc8a78",
        "#fe640b",
        "#df8e1d",
        "#d20f39",
        "#40a02b",
    ),
    "catppuccin-frappe": CatppuccinPalette(
        "#232634",
        "#303446",
        "#c6d0f5",
        "#a5adce",
        "#ca9ee6",
        "#ef9f76",
        "#f4b8e4",
        "#e5c890",
        "#e78284",
        "#a6d189",
    ),
    "catppuccin-macchiato": CatppuccinPalette(
        "#181926",
        "#24273a",
        "#cad3f5",
        "#a5adcb",
        "#c6a0f6",
        "#f5a97f",
        "#f5bde6",
        "#eed49f",
        "#ed8796",
        "#a6da95",
    ),
    "catppuccin-mocha": CatppuccinPalette(
        "#11111b",
        "#1e1e2e",
        "#cdd6f4",
        "#a6adc8",
        "#f5c2e7",
        "#cba6f7",
        "#fab387",
        "#fae3b0",
        "#f28fad",
        "#abe9b3",
    ),
}


def palette_for(theme: CatppuccinTheme) -> CatppuccinPalette:
    """Return the semantic palette for a validated Catppuccin theme."""
    return CATPPUCCIN_PALETTES[theme]


def css_variables_for(theme: CatppuccinTheme) -> dict[str, str]:
    """Return ModelTop-specific TCSS variables for a validated theme."""
    palette = palette_for(theme)
    return {
        "catppuccin-crust": palette.crust,
        "catppuccin-base": palette.base,
        "catppuccin-muted": palette.muted,
    }
