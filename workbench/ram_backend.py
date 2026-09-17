"""CPU-resident LBM collision with bounded JAX device batches.

``RamCollision`` is the bridge used by the GPU+RAM execution mode.  The
complete lattice state and macroscopic fields remain in host RAM.  For each
call, only one fixed-size batch is placed on ``gpu_device``; XLB's supplied
equilibrium and BGK collision operators run there, and the resulting
distribution is synchronously copied back before the next batch starts.

The operators are injected by the caller so this module has no XLB or JAX
import-time dependency.  Batches use channel-first FP32 arrays with channels
``f=27``, ``rho=1``, ``velocity=3`` and ``omega=1``.  When a forcing source is
configured and an acceleration field is supplied, acceleration adds a
three-channel FP32 batch.  The final batch is zero-padded to the fixed
``batch_cells`` shape and padding is discarded from the returned array.
``stats`` reports bytes for those padded device transfers, making the memory
traffic visible to the run diagnostics.  The ``batches``, ``upload_bytes``
and ``download_bytes`` fields describe the most recent call; ``total_batches``,
``total_upload_bytes`` and ``total_download_bytes`` accumulate successful
calls on the instance.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


DEFAULT_BATCH_CELLS = 65_536
_FP32_BYTES = np.dtype(np.float32).itemsize
_CHANNELS = {"f": 27, "rho": 1, "velocity": 3, "omega": 1, "acceleration": 3}


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _as_host_array(value: Any, label: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an array-like value") from exc


def _field(
    value: Any,
    *,
    channels: int,
    spatial_shape: tuple[int, ...],
    label: str,
    allow_channel_omission: bool = False,
) -> np.ndarray:
    """Validate one channel-first state field without making a full copy."""

    array = _as_host_array(value, label)
    expected = (channels, *spatial_shape)
    if array.shape == expected:
        return array
    if allow_channel_omission and array.shape == spatial_shape:
        # expand_dims is a view.  It lets callers pass scalar fields such as
        # omega with the natural spatial shape while preserving channel-first
        # input to XLB.
        return np.expand_dims(array, axis=0)
    raise ValueError(f"{label} must have shape {expected}, got {array.shape}")


def _flat_view(array: np.ndarray, channels: int, cells: int) -> np.ndarray | None:
    """Return a copy-free channel/cell view, or None for strided input."""

    # Calling reshape on a non-contiguous array may silently allocate a full
    # state-sized copy.  Check the flag first; strided inputs are gathered one
    # bounded batch at a time instead.
    if not array.flags.c_contiguous:
        return None
    return array.reshape((channels, cells), order="C")


def _copy_batch(
    destination: np.ndarray,
    array: np.ndarray,
    flat: np.ndarray | None,
    coordinates: tuple[np.ndarray, ...],
    start: int,
    stop: int,
) -> int:
    """Copy one host range into a fixed-size FP32 batch and pad its tail."""

    count = stop - start
    if flat is not None:
        source = flat[:, start:stop]
    else:
        # Advanced indexing materializes only this bounded batch.  It avoids a
        # full-field flattening copy for F-contiguous/transposed host inputs.
        source = array[(slice(None), *coordinates)]
    try:
        destination[:, :count] = source
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("state fields must contain values representable as FP32") from exc
    if count < destination.shape[1]:
        destination[:, count:] = 0.0
    return count


class RamCollision:
    """Run equilibrium plus BGK collision on bounded device batches.

    Parameters are intentionally dependency-injected.  ``jax`` and ``jnp``
    are the namespaces used by the caller's XLB setup; ``gpu_device`` is a
    JAX device object (a CPU device is suitable for deterministic tests).
    ``equilibrium`` must have the XLB signature ``(rho, velocity)`` and
    ``collision`` the signature ``(f, feq, omega)``.  An optional
    ``forcing_source`` receives ``(rho, velocity, acceleration)`` and returns
    a 27-channel source; it is added with the Guo prefactor
    ``1 - omega/2`` while the fused device operator is running.
    """

    def __init__(
        self,
        jax: Any,
        jnp: Any,
        gpu_device: Any,
        equilibrium: Callable[..., Any],
        collision: Callable[..., Any],
        batch_cells: int = DEFAULT_BATCH_CELLS,
        forcing_source: Callable[..., Any] | None = None,
    ) -> None:
        if not callable(equilibrium):
            raise TypeError("equilibrium must be callable")
        if not callable(collision):
            raise TypeError("collision must be callable")
        if forcing_source is not None and not callable(forcing_source):
            raise TypeError("forcing_source must be callable or None")
        if not hasattr(jax, "device_put") or not hasattr(jax, "device_get"):
            raise TypeError("jax must provide device_put and device_get")
        self.jax = jax
        self.jnp = jnp
        self.gpu_device = gpu_device
        self.equilibrium = equilibrium
        self.collision = collision
        self.batch_cells = _positive_integer(batch_cells, "batch_cells")
        self.forcing_source = forcing_source

        def eager(f_device: Any, rho_device: Any, velocity_device: Any, omega_device: Any) -> Any:
            feq_device = self.equilibrium(rho_device, velocity_device)
            return self.collision(f_device, feq_device, omega_device)

        self._eager = eager
        self._fused = eager
        self._jit_fused = False
        self._forced_eager: Callable[..., Any] | None = None
        self._forced_fused: Callable[..., Any] | None = None
        self._forcing_jit_fused = False
        self._calls = 0
        self._total_batches = 0
        self._total_cells = 0
        self._total_padding_cells = 0
        self._total_upload_bytes = 0
        self._total_download_bytes = 0
        jit = getattr(jax, "jit", None)
        if callable(jit):
            try:
                self._fused = jit(eager)
                self._jit_fused = True
            except Exception:
                # A caller may provide a lightweight JAX-compatible namespace
                # without jit, or an operator may require eager execution.
                self._fused = eager

        if forcing_source is not None:
            def forced_eager(
                f_device: Any,
                rho_device: Any,
                velocity_device: Any,
                omega_device: Any,
                acceleration_device: Any,
            ) -> Any:
                feq_device = self.equilibrium(rho_device, velocity_device)
                post_device = self.collision(f_device, feq_device, omega_device)
                source_device = self.forcing_source(
                    rho_device, velocity_device, acceleration_device
                )
                return post_device + (1.0 - 0.5 * omega_device) * source_device

            self._forced_eager = forced_eager
            self._forced_fused = forced_eager
            if callable(jit):
                try:
                    self._forced_fused = jit(forced_eager)
                    self._forcing_jit_fused = True
                except Exception:
                    # As with the unforced path, a dependency-injected
                    # namespace may expose a best-effort jit wrapper.
                    self._forced_fused = forced_eager

        self.stats: dict[str, Any] = {
            "batch_cells": self.batch_cells,
            "batches": 0,
            "cells": 0,
            "valid_cells": 0,
            "padded_cells": 0,
            "padding_cells": 0,
            "upload_bytes": 0,
            "download_bytes": 0,
            "output_bytes": 0,
            "jit_fused": self._jit_fused,
            "forcing_enabled": forcing_source is not None,
            "forcing_jit_fused": self._forcing_jit_fused,
            "acceleration_channels": 0,
            "forcing_upload_bytes": 0,
            "bytes_in_use": None,
            "peak_bytes_in_use": None,
            "calls": 0,
            "total_batches": 0,
            "total_cells": 0,
            "total_padding_cells": 0,
            "total_upload_bytes": 0,
            "total_download_bytes": 0,
        }

    def _reset_stats(self, cells: int, batches: int, *, forcing_enabled: bool) -> None:
        padded_cells = batches * self.batch_cells
        base_upload_channels = (
            _CHANNELS["f"]
            + _CHANNELS["rho"]
            + _CHANNELS["velocity"]
            + _CHANNELS["omega"]
        )
        acceleration_channels = _CHANNELS["acceleration"] if forcing_enabled else 0
        upload_channels = base_upload_channels + acceleration_channels
        download_channels = _CHANNELS["f"]
        forcing_upload_bytes = padded_cells * acceleration_channels * _FP32_BYTES
        self.stats = {
            "batch_cells": self.batch_cells,
            "batches": batches,
            "cells": cells,
            "valid_cells": cells,
            "padded_cells": padded_cells,
            "padding_cells": padded_cells - cells,
            "upload_bytes": padded_cells * upload_channels * _FP32_BYTES,
            "download_bytes": padded_cells * download_channels * _FP32_BYTES,
            "output_bytes": cells * download_channels * _FP32_BYTES,
            "jit_fused": self._jit_fused,
            "forcing_enabled": forcing_enabled,
            "forcing_jit_fused": self._forcing_jit_fused if forcing_enabled else False,
            "acceleration_channels": acceleration_channels,
            "forcing_upload_bytes": forcing_upload_bytes,
            "bytes_in_use": None,
            "peak_bytes_in_use": None,
            # The primary byte/batch fields describe this call.  Cumulative
            # counters are filled after a successful call below.
            "calls": self._calls,
            "total_batches": self._total_batches,
            "total_cells": self._total_cells,
            "total_padding_cells": self._total_padding_cells,
            "total_upload_bytes": self._total_upload_bytes,
            "total_download_bytes": self._total_download_bytes,
        }

    def _memory_stats(self) -> tuple[Any, Any]:
        """Read optional JAX allocator counters without making them required."""

        reader = getattr(self.gpu_device, "memory_stats", None)
        if not callable(reader):
            return None, None
        try:
            values = reader()
        except Exception:
            return None, None
        if not hasattr(values, "get"):
            return None, None
        try:
            return values.get("bytes_in_use"), values.get("peak_bytes_in_use")
        except Exception:
            return None, None

    def _device_array(self, batch: np.ndarray) -> Any:
        """Place an already-FP32 batch explicitly on the target device.

        ``jax.device_put`` accepts NumPy arrays directly.  Going through
        ``jnp.asarray`` first can create an unnecessary default-device buffer
        before the explicit transfer, so the injected ``jnp`` namespace is
        retained for operator compatibility but is intentionally not used
        here.
        """

        return self.jax.device_put(batch, self.gpu_device)

    def __call__(
        self,
        f: Any,
        rho: Any,
        velocity: Any,
        omega: Any,
        acceleration: Any | None = None,
    ) -> np.ndarray:
        """Return the collision result as a host ``np.float32`` array.

        The returned shape exactly matches ``f``.  Inputs are read only from
        this helper's perspective and are never modified.
        """

        f_array = _as_host_array(f, "f")
        if f_array.ndim < 1 or f_array.shape[0] != _CHANNELS["f"]:
            raise ValueError(f"f must have 27 leading channels, got {f_array.shape}")
        spatial_shape = tuple(f_array.shape[1:])
        cells = int(math.prod(spatial_shape)) if spatial_shape else 1
        if cells <= 0:
            raise ValueError("state fields must contain at least one cell")

        rho_array = _field(
            rho,
            channels=_CHANNELS["rho"],
            spatial_shape=spatial_shape,
            label="rho",
            allow_channel_omission=True,
        )
        velocity_array = _field(
            velocity,
            channels=_CHANNELS["velocity"],
            spatial_shape=spatial_shape,
            label="velocity",
        )
        omega_array = _field(
            omega,
            channels=_CHANNELS["omega"],
            spatial_shape=spatial_shape,
            label="omega",
            allow_channel_omission=True,
        )
        forcing_enabled = self.forcing_source is not None and acceleration is not None
        acceleration_array: np.ndarray | None = None
        if forcing_enabled:
            acceleration_array = _field(
                acceleration,
                channels=_CHANNELS["acceleration"],
                spatial_shape=spatial_shape,
                label="acceleration",
            )

        batches = (cells + self.batch_cells - 1) // self.batch_cells
        self._reset_stats(cells, batches, forcing_enabled=forcing_enabled)

        # These reshapes are views for the usual channel-first C-contiguous
        # solver state.  Strided arrays are handled by bounded advanced-index
        # gathers in _copy_batch and never flattened wholesale.
        f_flat = _flat_view(f_array, _CHANNELS["f"], cells)
        rho_flat = _flat_view(rho_array, _CHANNELS["rho"], cells)
        velocity_flat = _flat_view(velocity_array, _CHANNELS["velocity"], cells)
        omega_flat = _flat_view(omega_array, _CHANNELS["omega"], cells)
        acceleration_flat = (
            _flat_view(acceleration_array, _CHANNELS["acceleration"], cells)
            if acceleration_array is not None
            else None
        )

        # Keep writable 2-D host lines for cheap view reshapes to the 3-D
        # lattice layout expected by XLB.  Every device batch has fixed shape
        # (f:27,batch,1,1), (rho:1,batch,1,1), (velocity:3,batch,1,1), and
        # (omega:batch,1,1).  Only these bounded lines are filled each loop.
        f_line = np.empty((_CHANNELS["f"], self.batch_cells), dtype=np.float32)
        rho_line = np.empty((_CHANNELS["rho"], self.batch_cells), dtype=np.float32)
        velocity_line = np.empty((_CHANNELS["velocity"], self.batch_cells), dtype=np.float32)
        omega_line = np.empty((_CHANNELS["omega"], self.batch_cells), dtype=np.float32)
        acceleration_line = (
            np.empty((_CHANNELS["acceleration"], self.batch_cells), dtype=np.float32)
            if forcing_enabled
            else None
        )
        f_batch = f_line.reshape((_CHANNELS["f"], self.batch_cells, 1, 1), order="C")
        rho_batch = rho_line.reshape((_CHANNELS["rho"], self.batch_cells, 1, 1), order="C")
        velocity_batch = velocity_line.reshape((_CHANNELS["velocity"], self.batch_cells, 1, 1), order="C")
        omega_batch = omega_line.reshape((self.batch_cells, 1, 1), order="C")
        acceleration_batch = (
            acceleration_line.reshape((_CHANNELS["acceleration"], self.batch_cells, 1, 1), order="C")
            if acceleration_line is not None
            else None
        )
        f_post_flat = np.empty((_CHANNELS["f"], cells), dtype=np.float32)
        spatial_coordinates: tuple[np.ndarray, ...] = ()

        for batch_index in range(batches):
            start = batch_index * self.batch_cells
            stop = min(cells, start + self.batch_cells)
            if (
                f_flat is None
                or rho_flat is None
                or velocity_flat is None
                or omega_flat is None
                or (acceleration_flat is None and forcing_enabled)
            ):
                # np.unravel_index needs at least one spatial axis.  A 27
                # element 0-D-cell field is C-contiguous, so this branch only
                # occurs for genuinely strided fields.
                if not spatial_shape:
                    raise ValueError("0-D state fields must be contiguous")
                spatial_coordinates = tuple(
                    np.asarray(index, dtype=np.intp)
                    for index in np.unravel_index(np.arange(start, stop, dtype=np.intp), spatial_shape)
                )

            count = _copy_batch(f_line, f_array, f_flat, spatial_coordinates, start, stop)
            _copy_batch(rho_line, rho_array, rho_flat, spatial_coordinates, start, stop)
            _copy_batch(velocity_line, velocity_array, velocity_flat, spatial_coordinates, start, stop)
            _copy_batch(omega_line, omega_array, omega_flat, spatial_coordinates, start, stop)
            if forcing_enabled:
                # The guarded allocation above makes these values non-None;
                # keep the explicit check so an internal invariant failure is
                # reported before a device transfer.
                if acceleration_line is None or acceleration_array is None:
                    raise RuntimeError("internal forcing batch allocation error")
                _copy_batch(
                    acceleration_line,
                    acceleration_array,
                    acceleration_flat,
                    spatial_coordinates,
                    start,
                    stop,
                )
            # ``count`` is intentionally read to make it clear that only the
            # valid prefix is copied back below; padding never reaches output.
            if count != stop - start:  # pragma: no cover - defensive invariant
                raise RuntimeError("internal batch-size accounting error")

            f_device = rho_device = velocity_device = omega_device = None
            acceleration_device = post_device = None
            try:
                f_device = self._device_array(f_batch)
                rho_device = self._device_array(rho_batch)
                velocity_device = self._device_array(velocity_batch)
                omega_device = self._device_array(omega_batch)
                if forcing_enabled:
                    if acceleration_batch is None or self._forced_fused is None:
                        raise RuntimeError("internal forcing operator setup error")
                    acceleration_device = self._device_array(acceleration_batch)
                    post_device = self._forced_fused(
                        f_device,
                        rho_device,
                        velocity_device,
                        omega_device,
                        acceleration_device,
                    )
                else:
                    post_device = self._fused(f_device, rho_device, velocity_device, omega_device)
                post_host = np.asarray(self.jax.device_get(post_device))
                expected_batch_shape = (_CHANNELS["f"], self.batch_cells, 1, 1)
                if post_host.shape != expected_batch_shape:
                    raise ValueError(
                        f"collision returned shape {post_host.shape}; expected {expected_batch_shape}"
                    )
                try:
                    post_line = post_host.reshape((_CHANNELS["f"], self.batch_cells), order="C")
                    f_post_flat[:, start:stop] = post_line[:, :count]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("collision result must be convertible to FP32") from exc
            finally:
                # Drop device references before the next upload so memory
                # peaks at one batch plus the host-resident full state.
                del f_device, rho_device, velocity_device, omega_device, acceleration_device, post_device

        result = f_post_flat.reshape(f_array.shape, order="C")
        self._calls += 1
        self._total_batches += batches
        self._total_cells += cells
        self._total_padding_cells += batches * self.batch_cells - cells
        self._total_upload_bytes += int(self.stats["upload_bytes"])
        self._total_download_bytes += int(self.stats["download_bytes"])
        self.stats.update(
            calls=self._calls,
            total_batches=self._total_batches,
            total_cells=self._total_cells,
            total_padding_cells=self._total_padding_cells,
            total_upload_bytes=self._total_upload_bytes,
            total_download_bytes=self._total_download_bytes,
        )
        bytes_in_use, peak_bytes_in_use = self._memory_stats()
        self.stats["bytes_in_use"] = bytes_in_use
        self.stats["peak_bytes_in_use"] = peak_bytes_in_use
        return result


__all__ = ["DEFAULT_BATCH_CELLS", "RamCollision"]
