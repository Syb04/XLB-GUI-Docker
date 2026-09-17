import math
import unittest
from functools import partial

import numpy as np

from workbench.forcing import guo_source
from workbench.ram_backend import RamCollision


class _CpuDevice:
    def __init__(self, *, memory_stats=None):
        self._memory_stats = memory_stats

    def memory_stats(self):
        if self._memory_stats is None:
            raise NotImplementedError
        return self._memory_stats


class _NumpyJax:
    """Small CPU stand-in that records explicit device puts."""

    def __init__(self):
        self.puts = []
        self.jit_calls = 0

    def jit(self, function):
        self.jit_calls += 1
        return function

    def device_put(self, value, device):
        self.puts.append((value.shape, value.dtype, device))
        return np.array(value, copy=True)

    def device_get(self, value):
        return np.asarray(value)


class RamCollisionTests(unittest.TestCase):
    @staticmethod
    def _equilibrium(rho, velocity):
        q = np.arange(27, dtype=np.float32).reshape((27,) + (1,) * (rho.ndim - 1))
        return rho + q * np.float32(0.01) + velocity[0] * np.float32(0.1) + velocity[1] * np.float32(0.2) + velocity[2] * np.float32(0.3)

    @staticmethod
    def _collision(f, feq, omega):
        return f + omega * (feq - f)

    def _state(self, shape=(2, 3, 4)):
        rng = np.random.default_rng(902)
        f = rng.normal(size=(27, *shape)).astype(np.float32)
        rho = (1.0 + rng.random(size=(1, *shape))).astype(np.float32)
        velocity = (0.02 * rng.normal(size=(3, *shape))).astype(np.float32)
        omega = (0.4 + 0.2 * rng.random(size=shape)).astype(np.float32)
        return f, rho, velocity, omega

    def test_non_divisible_batches_match_reference_and_transfer_only_fixed_shapes(self):
        f, rho, velocity, omega = self._state()
        snapshots = tuple(value.copy() for value in (f, rho, velocity, omega))
        jax = _NumpyJax()
        device = _CpuDevice(memory_stats={"bytes_in_use": 1234, "peak_bytes_in_use": 5678})
        ram = RamCollision(
            jax,
            object(),
            device,
            self._equilibrium,
            self._collision,
            batch_cells=7,
        )

        actual = ram(f, rho, velocity, omega)
        expected = self._collision(f, self._equilibrium(rho, velocity), omega)

        self.assertEqual(actual.shape, f.shape)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
        for before, after in zip(snapshots, (f, rho, velocity, omega)):
            np.testing.assert_array_equal(after, before)

        self.assertEqual(ram.stats["batches"], 4)
        self.assertEqual(ram.stats["cells"], 24)
        self.assertEqual(ram.stats["padded_cells"], 28)
        self.assertEqual(ram.stats["padding_cells"], 4)
        self.assertEqual(ram.stats["upload_bytes"], 28 * (27 + 1 + 3 + 1) * 4)
        self.assertEqual(ram.stats["download_bytes"], 28 * 27 * 4)
        self.assertEqual(ram.stats["output_bytes"], 24 * 27 * 4)
        self.assertEqual(ram.stats["bytes_in_use"], 1234)
        self.assertEqual(ram.stats["peak_bytes_in_use"], 5678)
        self.assertEqual(ram.stats["calls"], 1)
        self.assertEqual(ram.stats["total_batches"], 4)
        self.assertEqual(ram.stats["total_upload_bytes"], ram.stats["upload_bytes"])
        self.assertEqual(ram.stats["total_download_bytes"], ram.stats["download_bytes"])

        expected_shapes = [
            (27, 7, 1, 1),
            (1, 7, 1, 1),
            (3, 7, 1, 1),
            (7, 1, 1),
        ] * 4
        self.assertEqual([shape for shape, _, _ in jax.puts], expected_shapes)
        self.assertTrue(all(dtype == np.dtype(np.float32) for _, dtype, _ in jax.puts))
        self.assertTrue(all(target is device for _, _, target in jax.puts))
        self.assertEqual(jax.jit_calls, 1)

    def test_one_cell_and_different_batch_sizes_match_reference(self):
        f, rho, velocity, omega = self._state(shape=(1,))
        expected = self._collision(f, self._equilibrium(rho, velocity), omega)
        for batch_cells in (1, 2, 5, 64):
            with self.subTest(batch_cells=batch_cells):
                ram = RamCollision(
                    _NumpyJax(),
                    object(),
                    _CpuDevice(),
                    self._equilibrium,
                    self._collision,
                    batch_cells=batch_cells,
                )
                actual = ram(f, rho, velocity, omega)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
                self.assertEqual(actual.shape, (27, 1))
                self.assertEqual(ram.stats["cells"], 1)
                self.assertEqual(ram.stats["batches"], 1)
                self.assertEqual(ram.stats["padding_cells"], batch_cells - 1)

    def test_strided_host_fields_are_gathered_per_batch_without_full_flattening(self):
        f, rho, velocity, omega = self._state(shape=(2, 3, 4))
        f = f[..., ::-1]
        rho = rho[..., ::-1]
        velocity = velocity[..., ::-1]
        omega = omega[..., ::-1]
        expected = self._collision(f, self._equilibrium(rho, velocity), omega)
        ram = RamCollision(
            _NumpyJax(),
            object(),
            _CpuDevice(),
            self._equilibrium,
            self._collision,
            batch_cells=5,
        )

        actual = ram(f, rho, velocity, omega)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
        self.assertEqual(ram.stats["batches"], math.ceil(24 / 5))

    def test_omega_without_leading_channel_and_cumulative_stats(self):
        f, rho, velocity, omega = self._state(shape=(2, 2))
        jax = _NumpyJax()
        ram = RamCollision(jax, object(), _CpuDevice(), self._equilibrium, self._collision, batch_cells=3)

        first = ram(f, rho, velocity, omega[None, ...])
        second = ram(f, rho, velocity, omega)

        np.testing.assert_allclose(first, second, rtol=0.0, atol=1.0e-7)
        self.assertEqual(ram.stats["calls"], 2)
        self.assertEqual(ram.stats["total_batches"], 2 * math.ceil(4 / 3))
        self.assertEqual(ram.stats["total_cells"], 8)

    def test_forcing_source_adds_acceleration_batch_and_prefactored_source(self):
        f, rho, velocity, omega = self._state(shape=(2, 3, 4))
        acceleration = np.asarray(
            [
                np.full((2, 3, 4), 0.01, dtype=np.float32),
                np.full((2, 3, 4), -0.02, dtype=np.float32),
                np.full((2, 3, 4), 0.03, dtype=np.float32),
            ]
        )
        snapshots = tuple(value.copy() for value in (f, rho, velocity, omega, acceleration))

        def source(rho_device, velocity_device, acceleration_device):
            q = np.arange(27, dtype=np.float32).reshape((27, 1, 1, 1))
            return q * np.float32(0.002) * acceleration_device[0:1] + np.float32(0.1) * acceleration_device[1:2]

        jax = _NumpyJax()
        ram = RamCollision(
            jax,
            object(),
            _CpuDevice(),
            self._equilibrium,
            self._collision,
            batch_cells=7,
            forcing_source=source,
        )

        actual = ram(f, rho, velocity, omega, acceleration)
        expected_source = source(rho, velocity, acceleration)
        expected = self._collision(f, self._equilibrium(rho, velocity), omega)
        expected = expected + (1.0 - np.float32(0.5) * omega) * expected_source

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
        self.assertEqual(actual.dtype, np.float32)
        for before, after in zip(snapshots, (f, rho, velocity, omega, acceleration)):
            np.testing.assert_array_equal(after, before)
        self.assertTrue(ram.stats["forcing_enabled"])
        self.assertTrue(ram.stats["forcing_jit_fused"])
        self.assertEqual(ram.stats["acceleration_channels"], 3)
        self.assertEqual(ram.stats["forcing_upload_bytes"], 28 * 3 * 4)
        self.assertEqual(ram.stats["upload_bytes"], 28 * (27 + 1 + 3 + 1 + 3) * 4)
        self.assertEqual(ram.stats["download_bytes"], 28 * 27 * 4)
        expected_shapes = [
            (27, 7, 1, 1),
            (1, 7, 1, 1),
            (3, 7, 1, 1),
            (7, 1, 1),
            (3, 7, 1, 1),
        ] * 4
        self.assertEqual([shape for shape, _, _ in jax.puts], expected_shapes)

    def test_forcing_source_without_acceleration_keeps_legacy_path(self):
        f, rho, velocity, omega = self._state(shape=(2, 3))
        jax = _NumpyJax()

        def source(*_):
            raise AssertionError("source must not run without acceleration")

        ram = RamCollision(
            jax,
            object(),
            _CpuDevice(),
            self._equilibrium,
            self._collision,
            batch_cells=4,
            forcing_source=source,
        )
        actual = ram(f, rho, velocity, omega)
        expected = self._collision(f, self._equilibrium(rho, velocity), omega)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
        self.assertFalse(ram.stats["forcing_enabled"])
        self.assertEqual(ram.stats["acceleration_channels"], 0)
        self.assertEqual(ram.stats["forcing_upload_bytes"], 0)
        self.assertEqual(ram.stats["upload_bytes"], 2 * 4 * (27 + 1 + 3 + 1) * 4)

    def test_acceleration_without_forcing_source_keeps_legacy_path(self):
        f, rho, velocity, omega = self._state(shape=(2,))
        acceleration = np.zeros_like(velocity)
        ram = RamCollision(
            _NumpyJax(),
            object(),
            _CpuDevice(),
            self._equilibrium,
            self._collision,
            batch_cells=2,
        )
        actual = ram(f, rho, velocity, omega, acceleration)
        expected = self._collision(f, self._equilibrium(rho, velocity), omega)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
        self.assertFalse(ram.stats["forcing_enabled"])
        self.assertEqual(ram.stats["acceleration_channels"], 0)
        self.assertEqual(ram.stats["upload_bytes"], 2 * (27 + 1 + 3 + 1) * 4)

    def test_invalid_shapes_and_batch_size_are_rejected(self):
        f, rho, velocity, omega = self._state(shape=(2,))
        for invalid in (0, -1, True, 1.5):
            with self.subTest(batch_cells=invalid), self.assertRaises(ValueError):
                RamCollision(_NumpyJax(), object(), _CpuDevice(), self._equilibrium, self._collision, invalid)
        ram = RamCollision(_NumpyJax(), object(), _CpuDevice(), self._equilibrium, self._collision, 2)
        with self.assertRaisesRegex(ValueError, "27 leading"):
            ram(f[:26], rho, velocity, omega)
        with self.assertRaisesRegex(ValueError, "rho.*shape"):
            ram(f, rho[:, :1], velocity, omega)
        with self.assertRaisesRegex(ValueError, "velocity.*shape"):
            ram(f, rho, velocity[:2], omega)


class XlbRamCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import jax
            import jax.numpy as jnp
            import xlb
            from xlb import ComputeBackend, PrecisionPolicy
            from xlb.operator.collision import BGK
            from xlb.operator.equilibrium import QuadraticEquilibrium
        except ImportError as exc:  # pragma: no cover - exercised in WSL CI
            raise unittest.SkipTest(f"JAX/XLB unavailable: {exc}")
        policy = PrecisionPolicy.FP32FP32
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            velocity_set = xlb.velocity_set.D3Q27(
                precision_policy=policy,
                compute_backend=ComputeBackend.JAX,
            )
            xlb.init(
                velocity_set=velocity_set,
                default_backend=ComputeBackend.JAX,
                default_precision_policy=policy,
            )
            cls.velocity_set = velocity_set
            cls.equilibrium = QuadraticEquilibrium(
                velocity_set=velocity_set,
                precision_policy=policy,
                compute_backend=ComputeBackend.JAX,
            )
            cls.collision = BGK(
                velocity_set=velocity_set,
                precision_policy=policy,
                compute_backend=ComputeBackend.JAX,
            )
        cls.jax = jax
        cls.jnp = jnp
        cls.cpu = cpu

    def test_real_xlb_operators_match_full_cpu_reference(self):
        rng = np.random.default_rng(77)
        shape = (2, 3, 4)
        f = rng.random((27, *shape), dtype=np.float32)
        rho = (1.0 + 0.1 * rng.random((1, *shape))).astype(np.float32)
        velocity = (0.01 * rng.random((3, *shape))).astype(np.float32)
        omega = (0.5 + 0.1 * rng.random(shape)).astype(np.float32)
        ram = RamCollision(
            self.jax,
            self.jnp,
            self.cpu,
            self.equilibrium,
            self.collision,
            batch_cells=7,
        )

        actual = ram(f, rho, velocity, omega)
        with self.jax.default_device(self.cpu):
            f_device = self.jnp.asarray(f, dtype=self.jnp.float32)
            rho_device = self.jnp.asarray(rho, dtype=self.jnp.float32)
            velocity_device = self.jnp.asarray(velocity, dtype=self.jnp.float32)
            omega_device = self.jnp.asarray(omega, dtype=self.jnp.float32)
            expected_device = self.collision(
                f_device,
                self.equilibrium(rho_device, velocity_device),
                omega_device,
            )
        expected = np.asarray(self.jax.device_get(expected_device), dtype=np.float32)

        self.assertEqual(actual.shape, f.shape)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-6)
        self.assertEqual(ram.stats["batches"], 4)
        self.assertEqual(ram.stats["padding_cells"], 4)

    def test_real_xlb_forced_batches_match_full_cpu_reference(self):
        rng = np.random.default_rng(78)
        shape = (2, 3, 4)
        f = rng.random((27, *shape), dtype=np.float32)
        rho = (1.0 + 0.1 * rng.random((1, *shape))).astype(np.float32)
        velocity = (0.01 * rng.random((3, *shape))).astype(np.float32)
        omega = (0.5 + 0.1 * rng.random(shape)).astype(np.float32)
        acceleration = (0.001 * rng.normal(size=(3, *shape))).astype(np.float32)

        source = partial(
            guo_source,
            jnp=self.jnp,
            c=self.velocity_set.c,
            w=self.velocity_set.w,
        )

        ram = RamCollision(
            self.jax,
            self.jnp,
            self.cpu,
            self.equilibrium,
            self.collision,
            batch_cells=7,
            forcing_source=source,
        )
        actual = ram(f, rho, velocity, omega, acceleration)

        with self.jax.default_device(self.cpu):
            f_device = self.jnp.asarray(f, dtype=self.jnp.float32)
            rho_device = self.jnp.asarray(rho, dtype=self.jnp.float32)
            velocity_device = self.jnp.asarray(velocity, dtype=self.jnp.float32)
            omega_device = self.jnp.asarray(omega, dtype=self.jnp.float32)
            acceleration_device = self.jnp.asarray(acceleration, dtype=self.jnp.float32)
            expected_device = self.collision(
                f_device,
                self.equilibrium(rho_device, velocity_device),
                omega_device,
            )
            expected_device = expected_device + (
                1.0 - self.jnp.float32(0.5) * omega_device
            ) * source(rho_device, velocity_device, acceleration_device)
        expected = np.asarray(self.jax.device_get(expected_device), dtype=np.float32)

        self.assertEqual(actual.shape, f.shape)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-6)
        self.assertTrue(ram.stats["forcing_enabled"])
        self.assertEqual(ram.stats["acceleration_channels"], 3)
        self.assertEqual(ram.stats["upload_bytes"], 28 * (27 + 1 + 3 + 1 + 3) * 4)
        self.assertEqual(ram.stats["download_bytes"], 28 * 27 * 4)


if __name__ == "__main__":
    unittest.main()
