"""
Color scheme constants for consistent UI styling.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ColorScheme:
    """Consistent color scheme for the entire application."""
    PRIMARY: str = "#4f46e5"  # Indigo - main brand color
    PRIMARY_LIGHT: str = "#818cf8"  # Lighter indigo
    SECONDARY: str = "#10b981"  # Emerald green
    ACCENT: str = "#f59e0b"  # Amber
    BACKGROUND: str = "#f8fafc"  # Slate 50
    SURFACE: str = "#ffffff"  # White
    TEXT_PRIMARY: str = "#1e293b"  # Slate 800
    TEXT_SECONDARY: str = "#64748b"  # Slate 500
    BORDER: str = "#e2e8f0"  # Slate 200
    SUCCESS: str = "#10b981"  # Green
    WARNING: str = "#f59e0b"  # Amber
    ERROR: str = "#ef4444"  # Red

    # Chart colors
    CHART_COLORS: tuple = (
        "#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
        "#06b6d4", "#84cc16", "#f97316", "#ec4899", "#6366f1"
    )


# Create singleton instance
color_scheme = ColorScheme()
