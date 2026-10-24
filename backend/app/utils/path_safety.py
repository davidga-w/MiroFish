"""
Guards for turning client-supplied identifiers into filesystem paths.

Project, simulation and report IDs arrive in URLs and JSON bodies and are joined
onto storage directories under uploads/. Every ID the backend generates is made
of letters, digits, "_" and "-" (proj_<hex>, sim_<hex>, report_<hex>, task UUIDs),
so anything else, such as "..", "/", "\\" or NUL, is rejected before it can reach
a path. Without this, e.g. DELETE /api/report/.. removed the whole uploads/ tree.
"""

import os
import re

# Request fields whose values are joined onto storage directories.
PATH_ID_FIELDS = ("project_id", "simulation_id", "report_id")

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


class UnsafePathError(ValueError):
    """An identifier that could resolve outside its storage directory."""


def is_safe_id(value) -> bool:
    """True if `value` is a single, traversal-free path component."""
    return isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None


def safe_child_path(base_dir: str, child_id: str) -> str:
    """`base_dir/child_id`, refusing IDs that could resolve outside `base_dir`."""
    if not is_safe_id(child_id):
        raise UnsafePathError(f"Invalid identifier: {child_id!r}")
    return os.path.join(base_dir, child_id)
