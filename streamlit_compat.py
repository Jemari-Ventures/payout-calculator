"""Streamlit API compatibility across local and Cloud Streamlit versions."""
from __future__ import annotations

import inspect
from typing import Any, Dict

import streamlit as st


def _supports_param(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def stretch_width_kwargs(func) -> Dict[str, Any]:
    """Full-width kwarg for charts, dataframes, buttons, etc."""
    if _supports_param(func, "width"):
        return {"width": "stretch"}
    if _supports_param(func, "use_container_width"):
        return {"use_container_width": True}
    return {}


def render_html(html: str, *, height: int = 1200, scrolling: bool = True) -> None:
    """Render inline HTML; falls back to components.html when iframe lacks srcdoc."""
    if hasattr(st, "iframe"):
        try:
            st.iframe(srcdoc=html, height=height, scrolling=scrolling)
            return
        except TypeError:
            pass
    st.components.v1.html(html, height=height, scrolling=scrolling)
