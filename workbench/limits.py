"""Server allocation limits, read once at startup (also used by CLI clients)."""
import os


def positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0 or value > 2**31 - 1:
        raise ValueError(f"{name} must be between 1 and {2**31 - 1}")
    return value


MAX_TOTAL_CELLS = positive_env('XLB_MAX_TOTAL_CELLS', 10_000_000)
MAX_AXIS_CELLS = positive_env('XLB_MAX_AXIS_CELLS', 1024)
MAX_BOUNDARY_LINK_BYTES = positive_env('XLB_MAX_BOUNDARY_LINK_MIB', 512) * 1024 * 1024


def configured_limits():
    return {'max_total_cells': MAX_TOTAL_CELLS, 'max_axis_cells': MAX_AXIS_CELLS,
            'max_boundary_link_mib': MAX_BOUNDARY_LINK_BYTES // (1024 * 1024)}
