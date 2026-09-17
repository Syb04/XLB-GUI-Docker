"""Dependency-free Guo forcing helpers for D3Q27 lattice Boltzmann runs.

The functions in this module deliberately do not import JAX, XLB, or NumPy
at module import time.  The array namespace is supplied by the caller so the
same implementation can be used with NumPy on the host and ``jax.numpy`` in
a compiled XLB operator.  ``velocity`` is the velocity used by the discrete
forcing expression (the physical, half-force-corrected velocity when the
caller keeps a separate raw velocity).

For a body acceleration ``a`` the force is ``F = rho*a`` and the Guo source is

``S_q = w_q * (3*(c_q-u) + 9*(c_q.u)*c_q) . F``.

The returned arrays are always FP32.  Inputs are converted to temporary
arrays and are never modified.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


_Q = 27
_D = 3


def _float32(jnp: Any) -> Any:
    """Return the FP32 dtype exposed by NumPy-like array namespaces."""

    return getattr(jnp, "float32", np.float32)


def _array(jnp: Any, value: Any) -> Any:
    """Convert a value to an FP32 array through the injected namespace."""

    asarray = getattr(jnp, "asarray", None)
    if not callable(asarray):
        raise TypeError("jnp must provide asarray")
    try:
        return asarray(value, dtype=_float32(jnp))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("forcing inputs must be numeric array-like values") from exc


def _validate_inputs(jnp: Any, rho: Any, velocity: Any, acceleration: Any, c: Any, w: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Convert and validate source inputs, returning broadcast-ready arrays."""

    rho_array = _array(jnp, rho)
    velocity_array = _array(jnp, velocity)
    acceleration_array = _array(jnp, acceleration)
    c_array = _array(jnp, c)
    w_array = _array(jnp, w)

    if velocity_array.ndim < 1 or velocity_array.shape[0] != _D:
        raise ValueError(
            f"velocity must have shape (3, ...), got {velocity_array.shape}"
        )
    if acceleration_array.shape != velocity_array.shape:
        raise ValueError(
            "acceleration must have the same shape as velocity; "
            f"got {acceleration_array.shape} and {velocity_array.shape}"
        )
    if c_array.ndim != 2 or c_array.shape != (_D, _Q):
        raise ValueError(f"c must have shape (3, 27), got {c_array.shape}")
    if w_array.ndim != 1 or w_array.shape != (_Q,):
        raise ValueError(f"w must have shape (27,), got {w_array.shape}")

    spatial_shape = tuple(velocity_array.shape[1:])
    if rho_array.ndim == 0:
        rho_scalar = rho_array
    elif rho_array.ndim == velocity_array.ndim and rho_array.shape[0] == 1:
        rho_scalar = rho_array[0]
    elif rho_array.ndim == velocity_array.ndim - 1:
        rho_scalar = rho_array
    else:
        raise ValueError(
            "rho must be scalar, have shape (N, ...), or have shape (1, N, ...); "
            f"got {rho_array.shape} for velocity shape {velocity_array.shape}"
        )

    # Check broadcastability using only static shapes.  This does not touch
    # array values and remains safe when the arrays are JAX tracers under jit.
    try:
        broadcast = np.broadcast_shapes(tuple(rho_scalar.shape), spatial_shape)
    except ValueError as exc:
        raise ValueError(
            f"rho shape {rho_array.shape} is not broadcastable to velocity shape "
            f"{velocity_array.shape}"
        ) from exc
    if broadcast != spatial_shape:
        raise ValueError(
            f"rho shape {rho_array.shape} is not broadcastable to velocity shape "
            f"{velocity_array.shape}"
        )

    return rho_scalar, velocity_array, acceleration_array, c_array, w_array


def guo_source(
    rho: Any,
    velocity: Any,
    acceleration: Any,
    *,
    jnp: Any,
    c: Any,
    w: Any,
) -> Any:
    """Return the D3Q27 Guo source term in FP32.

    Parameters use channel-first layouts: ``rho`` is ``(1, *shape)`` (a
    scalar ``(*shape)`` or a scalar value is also accepted), while
    ``velocity`` and ``acceleration`` are ``(3, *shape)``.  ``c`` must be
    ``(3, 27)`` and ``w`` must be ``(27,)``.  The result has shape
    ``(27, *shape)``.

    ``velocity`` is used exactly as supplied in the source expression.  The
    caller is responsible for passing the physical half-force-corrected
    velocity when that convention is used by the surrounding solver.
    """

    rho_scalar, velocity_array, acceleration_array, c_array, w_array = _validate_inputs(
        jnp, rho, velocity, acceleration, c, w
    )
    spatial_shape = tuple(velocity_array.shape[1:])
    ones = (1,) * len(spatial_shape)

    # q is the leading result axis.  Contracting the component axis directly
    # avoids materialising a ``(3, 27, *shape)`` temporary for a full CPU
    # domain.  ``einsum`` is lowered/fused by JAX and NumPy keeps only the
    # q-sized result here.
    w_q = w_array.reshape((_Q, *ones))
    force = rho_scalar * acceleration_array
    c_dot_u = jnp.einsum("iq,i...->q...", c_array, velocity_array)
    c_dot_force = jnp.einsum("iq,i...->q...", c_array, force)
    u_dot_force = jnp.sum(velocity_array * force, axis=0)
    c_minus_u_dot_force = c_dot_force - u_dot_force
    source = w_q * (3.0 * c_minus_u_dot_force + 9.0 * c_dot_u * c_dot_force)
    return _array(jnp, source)


def corrected_equilibrium(
    eq: Callable[..., Any],
    rho: Any,
    u: Any,
    a: Any,
    *,
    jnp: Any,
    c: Any,
    w: Any,
) -> Any:
    """Return ``eq(rho, u) - 0.5*guo_source(rho, u, a)`` in FP32.

    This half-source correction is useful for boundary and initial states
    where the prescribed ``u`` is the physical velocity.  The equilibrium
    callable is injected so this helper remains independent of XLB.
    """

    if not callable(eq):
        raise TypeError("eq must be callable")
    source = guo_source(rho, u, a, jnp=jnp, c=c, w=w)
    base = _array(jnp, eq(rho, u))
    return _array(jnp, base - 0.5 * source)


__all__ = ["corrected_equilibrium", "guo_source"]
