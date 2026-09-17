import unittest

import numpy as np

from workbench.residuals import ConvergenceMonitor, make_residual_operator


class ResidualOperatorTests(unittest.TestCase):
    def test_monitor_does_not_mutate_float64_input_or_readonly_views(self):
        fields = list(self._fields())
        for index in range(6):
            fields[index] = fields[index].astype(np.float64)
            fields[index].flags.writeable = False
        before = [value.copy() for value in fields]
        operator = make_residual_operator(np, initial_temperature=300., velocity_scale=5.,
                                         pressure_scale=100., flow_enabled=True, thermal_enabled=True)
        a = operator(*fields)
        b = operator(*fields)
        np.testing.assert_array_equal(a, b)
        for actual, expected in zip(fields, before):
            np.testing.assert_array_equal(actual, expected)

    def _fields(self):
        shape = (2, 2, 1)
        current_velocity = np.zeros((3, *shape), dtype=np.float32)
        previous_velocity = np.zeros_like(current_velocity)
        # One active fluid cell changes by 0.5 m/s with a scale of 2 m/s per
        # lattice unit.  The inactive cell contains a deliberately large
        # value to prove that the mask participates in every reduction.
        current_velocity[:, 0, 0, 0] = [0.15, 0.20, 0.0]
        current_velocity[:, 1, 0, 0] = [100.0, 100.0, 100.0]
        previous_velocity[:, 1, 0, 0] = [-100.0, -100.0, -100.0]

        current_rho = np.ones((1, *shape), dtype=np.float32)
        previous_rho = np.ones_like(current_rho)
        current_rho[0, 0, 0, 0] = 1.20
        current_rho[0, 1, 0, 0] = 100.0
        previous_rho[0, 1, 0, 0] = -100.0

        current_temperature = np.full(shape, 300.0, dtype=np.float32)
        previous_temperature = current_temperature.copy()
        # Thermal masks include solid cells: this cell is not fluid, but its
        # temperature change is still part of the thermal residual.
        current_temperature[1, 0, 0] = 302.0
        previous_temperature[1, 0, 0] = 301.0
        current_temperature[0, 1, 0] = 10000.0
        previous_temperature[0, 1, 0] = -10000.0

        fluid = np.asarray([[[True], [False]], [[False], [False]]]).reshape(shape)
        thermal = np.asarray([[[True], [False]], [[True], [False]]]).reshape(shape)
        return (
            current_velocity,
            current_rho,
            current_temperature,
            previous_velocity,
            previous_rho,
            previous_temperature,
            fluid,
            thermal,
        )

    def test_physical_normalization_and_separate_masks(self):
        operator = make_residual_operator(
            np,
            initial_temperature=300.0,
            velocity_scale=2.0,
            pressure_scale=10.0,
            flow_enabled=True,
            thermal_enabled=True,
        )
        result = operator(*self._fields())

        self.assertEqual(result.shape, (6,))
        self.assertEqual(result.dtype, np.float64)
        # v delta = 0.5 m/s and current max speed = 0.5 m/s.
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[3], 0.5)
        # p delta = 2 Pa and current max |p| = 2 Pa.
        np.testing.assert_allclose(result[1], 1.0)
        # Input density is FP32: 1.2 is not exactly representable.
        np.testing.assert_allclose(result[4], (float(np.float32(1.2)) - 1.0) * 10.0)
        # T delta = 1 K and max deviation from the 300 K initial state is 2 K.
        np.testing.assert_allclose(result[2], 0.5)
        np.testing.assert_allclose(result[5], 1.0)

    def test_temperature_is_not_normalized_by_absolute_temperature(self):
        shape = (1, 1, 1)
        current_temperature = np.full(shape, 301.0, dtype=np.float32)
        previous_temperature = np.full(shape, 300.0, dtype=np.float32)
        zeros_velocity = np.zeros((3, *shape), dtype=np.float32)
        ones_rho = np.ones((1, *shape), dtype=np.float32)
        mask = np.ones(shape, dtype=bool)
        operator = make_residual_operator(
            np,
            initial_temperature=300.0,
            velocity_scale=1.0,
            pressure_scale=1.0,
            flow_enabled=False,
            thermal_enabled=True,
        )
        result = operator(
            zeros_velocity,
            ones_rho,
            current_temperature,
            zeros_velocity,
            ones_rho,
            previous_temperature,
            mask,
            mask,
        )
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    def test_zero_baseline_and_disabled_physics_return_fp64_zeros(self):
        shape = (2, 1, 1)
        velocity = np.full((3, *shape), 4.0, dtype=np.float32)
        rho = np.full((1, *shape), 2.0, dtype=np.float32)
        temperature = np.full(shape, 500.0, dtype=np.float32)
        mask = np.ones(shape, dtype=bool)

        enabled = make_residual_operator(
            np,
            initial_temperature=300.0,
            velocity_scale=3.0,
            pressure_scale=100.0,
            flow_enabled=True,
            thermal_enabled=True,
        )
        result = enabled(velocity, rho, temperature, velocity, rho, temperature, mask, mask)
        np.testing.assert_array_equal(result, np.zeros(6, dtype=np.float64))

        disabled = make_residual_operator(
            np,
            initial_temperature=300.0,
            velocity_scale=3.0,
            pressure_scale=100.0,
            flow_enabled=False,
            thermal_enabled=False,
        )
        result = disabled(velocity, rho, temperature, np.zeros_like(velocity), np.ones_like(rho), np.zeros_like(temperature), mask, mask)
        np.testing.assert_array_equal(result, np.zeros(6, dtype=np.float64))

    def test_shape_errors_are_explicit(self):
        shape = (2, 1, 1)
        velocity = np.zeros((3, *shape), dtype=np.float32)
        rho = np.ones((1, *shape), dtype=np.float32)
        temperature = np.ones(shape, dtype=np.float32)
        mask = np.ones(shape, dtype=bool)
        operator = make_residual_operator(
            np,
            initial_temperature=300.0,
            velocity_scale=1.0,
            pressure_scale=1.0,
            flow_enabled=True,
            thermal_enabled=True,
        )
        with self.assertRaisesRegex(ValueError, "current_rho"):
            operator(velocity, rho[0], temperature, velocity, rho, temperature, mask, mask)


class ConvergenceMonitorTests(unittest.TestCase):
    def test_baseline_exact_intervals_threshold_and_sample_metadata(self):
        monitor = ConvergenceMonitor(
            tolerance=1.0e-3,
            consecutive_samples=3,
            interval=20,
            dt=0.01,
            flow_enabled=True,
            thermal_enabled=True,
        )
        values = [1.0e-3, 2.0e-4, 3.0e-4, 1.0, 2.0, 3.0]

        baseline = monitor.update(0)
        self.assertIsNone(baseline["residual"])
        self.assertIsNone(baseline["sample_steps"])
        self.assertFalse(baseline["monitor_eligible"])
        first = monitor.update(20, values)
        self.assertEqual(first["residual"], 1.0e-3)
        self.assertEqual(first["monitor_consecutive_samples"], 1)
        second = monitor.update(40, values)
        self.assertEqual(second["sample_steps"], 20)
        self.assertAlmostEqual(second["sample_time"], 0.2)
        self.assertEqual(second["residual"], 1.0e-3)
        self.assertEqual(second["monitor_consecutive_samples"], 2)
        self.assertTrue(second["monitor_eligible"])
        third = monitor.update(60, values)
        self.assertEqual(third["monitor_consecutive_samples"], 3)
        self.assertTrue(third["monitor_satisfied"])
        fourth = monitor.update(80, values)
        self.assertEqual(fourth["monitor_consecutive_samples"], 4)
        self.assertTrue(fourth["monitor_satisfied"])
        self.assertTrue(monitor.satisfied)

    def test_partial_gap_failure_and_fresh_baseline_restart_streak(self):
        monitor = ConvergenceMonitor(consecutive_samples=2, interval=20, dt=0.5)
        values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        monitor.update(0)
        monitor.update(20, values)
        passing = monitor.update(40, values)
        self.assertEqual(passing["monitor_consecutive_samples"], 2)
        partial = monitor.update(50, values)
        self.assertEqual(partial["sample_steps"], 10)
        self.assertEqual(partial["monitor_consecutive_samples"], 0)
        self.assertFalse(partial["monitor_satisfied"])
        monitor.update(70, values)
        satisfied = monitor.update(90, values)
        self.assertTrue(satisfied["monitor_satisfied"])

        baseline = monitor.update(100)
        self.assertFalse(baseline["monitor_satisfied"])
        self.assertEqual(baseline["monitor_consecutive_samples"], 0)
        self.assertEqual(monitor.update(120, values)["monitor_consecutive_samples"], 1)
        fresh = monitor.update(140, values)
        self.assertEqual(fresh["monitor_consecutive_samples"], 2)
        self.assertTrue(fresh["monitor_satisfied"])

    def test_pressure_only_drift_and_inactive_fields(self):
        monitor = ConvergenceMonitor(
            tolerance=1.0e-5,
            consecutive_samples=1,
            interval=20,
            flow_enabled=True,
            thermal_enabled=False,
        )
        monitor.update(0)
        values = [0.0, 2.0e-5, 9.0, 0.0, 2.0, 9.0]
        monitor.update(20, values)
        row = monitor.update(40, values)
        self.assertEqual(row["residual"], 2.0e-5)
        self.assertFalse(row["monitor_satisfied"])
        self.assertIsNone(row["temperature_residual"])
        self.assertIsNone(row["temperature_change_max"])

    def test_no_physics_never_satisfies_and_nonfinite_active_values_rejected(self):
        monitor = ConvergenceMonitor(consecutive_samples=1, interval=1, flow_enabled=False, thermal_enabled=False)
        monitor.update(0)
        monitor.update(1, [0.0] * 6)
        row = monitor.update(2, [0.0] * 6)
        self.assertIsNone(row["residual"])
        self.assertFalse(row["monitor_eligible"])
        self.assertFalse(row["monitor_satisfied"])

        flow_monitor = ConvergenceMonitor(flow_enabled=True, thermal_enabled=False)
        flow_monitor.update(0)
        with self.assertRaisesRegex(ValueError, "finite"):
            flow_monitor.update(1, [np.nan, 0.0, np.inf, 0.0, 0.0, np.nan])

        thermal_monitor = ConvergenceMonitor(flow_enabled=False, thermal_enabled=True)
        thermal_monitor.update(0)
        # Non-finite values in disabled flow slots are ignored.
        thermal_monitor.update(1, [np.nan, np.inf, 0.0, np.nan, np.inf, 0.0])

    def test_diagnostics_has_stable_definition_and_scales(self):
        monitor = ConvergenceMonitor()
        diagnostics = monitor.diagnostics()
        self.assertEqual(diagnostics["definition"], "sample_change_linf_v1")
        self.assertEqual(diagnostics["floors"], {"velocity_SI": 1.0e-6, "pressure_Pa": 1.0, "temperature_K": 1.0})
        self.assertEqual(diagnostics["required_samples"], 5)
        self.assertEqual(diagnostics["consecutive_samples"], 0)


class OptionalJaxParityTests(unittest.TestCase):
    def test_jax_jit_matches_numpy_when_available(self):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            self.skipTest("JAX is optional for this focused test")

        jax.config.update("jax_enable_x64", True)
        rng = np.random.default_rng(1234)
        shape = (2, 2, 2)
        current_velocity = rng.normal(size=(3, *shape)).astype(np.float32) * 0.01
        previous_velocity = rng.normal(size=(3, *shape)).astype(np.float32) * 0.01
        current_rho = (1.0 + rng.normal(size=(1, *shape)).astype(np.float32) * 1.0e-3)
        previous_rho = (1.0 + rng.normal(size=(1, *shape)).astype(np.float32) * 1.0e-3)
        current_temperature = (300.0 + rng.normal(size=shape)).astype(np.float32)
        previous_temperature = (300.0 + rng.normal(size=shape)).astype(np.float32)
        fluid = rng.random(shape) > 0.2
        thermal = rng.random(shape) > 0.1
        args = (
            current_velocity,
            current_rho,
            current_temperature,
            previous_velocity,
            previous_rho,
            previous_temperature,
            fluid,
            thermal,
        )
        kwargs = dict(
            initial_temperature=300.0,
            velocity_scale=0.2,
            pressure_scale=1500.0,
            flow_enabled=True,
            thermal_enabled=True,
        )
        numpy_result = make_residual_operator(np, **kwargs)(*args)
        jax_result = jax.jit(make_residual_operator(jnp, **kwargs))(
            *(jnp.asarray(value) for value in args)
        )
        self.assertEqual(jax_result.dtype, jnp.float64)
        np.testing.assert_allclose(np.asarray(jax_result), numpy_result, rtol=1.0e-11, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
