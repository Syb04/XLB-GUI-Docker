"""Parity and tracing checks for the small GPU field factories."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from workbench.fields_jax import make_les_operator, make_material_operator
from workbench.solver import _material_fields, _smagorinsky_eddy_viscosity


_HAS_JAX = importlib.util.find_spec("jax") is not None
if _HAS_JAX:
    import jax
    import jax.numpy as jnp


@unittest.skipUnless(_HAS_JAX, "the pinned WSL JAX environment is required")
class MaterialOperatorTests(unittest.TestCase):
    def test_constants_and_inactive_cells_match_cpu_solver(self) -> None:
        materials = [
            {
                "id": "water",
                "density": {"kind": "constant", "value": 998.0},
                "viscosity": {"kind": "constant", "value": 1.0e-3},
                "heat_capacity": {"kind": "constant", "value": 4182.0},
                "conductivity": {"kind": "constant", "value": 0.6},
            },
            # Missing properties intentionally exercise solver defaults.
            {"id": "air", "density": {"kind": "constant", "value": 1.2}},
        ]
        material_index = np.asarray([[0, 1, -1], [1, 0, 1]], dtype=np.int32)
        active = material_index >= 0
        temperature = np.asarray([[300.0, 301.0, 900.0], [302.0, 303.0, 304.0]])
        warnings: list[str] = []
        expected = _material_fields(materials, material_index, temperature, active, warnings)

        evaluate = make_material_operator(materials, material_index, active, jax=jax, jnp=jnp)
        actual = evaluate(temperature)
        for expected_field, actual_field in zip(expected, actual[:4]):
            np.testing.assert_allclose(np.asarray(actual_field), expected_field)
        self.assertFalse(bool(np.asarray(actual[4])))
        np.testing.assert_allclose(np.asarray(actual[0])[~active], 1.0)
        np.testing.assert_allclose(np.asarray(actual[1])[~active], 1.0)

    def test_multimaterial_tables_clamp_only_active_cells(self) -> None:
        materials = [
            {
                "id": "cold",
                "density": {"kind": "table", "points": [[280.0, 900.0], [320.0, 1000.0]]},
                "viscosity": {"kind": "table", "points": [[280.0, 0.8e-3], [320.0, 1.2e-3]]},
                "heat_capacity": {"kind": "constant", "value": 4000.0},
                "conductivity": {"kind": "constant", "value": 0.5},
            },
            {
                "id": "hot",
                "density": {"kind": "table", "points": [[300.0, 1000.0], [400.0, 800.0]]},
                "viscosity": {"kind": "constant", "value": 2.0e-3},
                "heat_capacity": {"kind": "constant", "value": 4100.0},
                "conductivity": {"kind": "constant", "value": 0.7},
            },
        ]
        material_index = np.asarray([[0, 1, -1], [1, 0, 1]], dtype=np.int32)
        active = material_index >= 0
        temperature = np.asarray([[250.0, 450.0, 250.0], [350.0, 325.0, 275.0]])
        warnings: list[str] = []
        expected = _material_fields(materials, material_index, temperature, active, warnings)

        evaluate = make_material_operator(materials, material_index, active, jax=jax, jnp=jnp)
        actual = evaluate(temperature)
        for expected_field, actual_field in zip(expected, actual[:4]):
            np.testing.assert_allclose(np.asarray(actual_field), expected_field)
        self.assertTrue(bool(np.asarray(actual[4])))

        in_range = np.asarray([[300.0, 350.0, 280.0], [350.0, 300.0, 320.0]])
        self.assertFalse(bool(np.asarray(evaluate(in_range)[4])))

    def test_material_operator_jits_and_returns_device_boolean_scalar(self) -> None:
        materials = [{"density": {"kind": "constant", "value": 2.0}}]
        material_index = np.zeros((2, 2, 2), dtype=np.int32)
        active = np.ones_like(material_index, dtype=bool)
        evaluate = make_material_operator(materials, material_index, active, jax=jax, jnp=jnp)
        density, viscosity, cp, conductivity, clamped = jax.jit(evaluate)(jnp.full((2, 2, 2), 300.0))
        self.assertEqual(density.shape, (2, 2, 2))
        self.assertEqual(clamped.shape, ())
        self.assertFalse(bool(np.asarray(clamped)))
        np.testing.assert_allclose(np.asarray(density), 2.0)
        np.testing.assert_allclose(np.asarray(viscosity), 1.0e-3)
        np.testing.assert_allclose(np.asarray(cp), 4182.0)
        np.testing.assert_allclose(np.asarray(conductivity), 0.6)

    def test_invalid_table_is_rejected_before_evaluation(self) -> None:
        with self.assertRaises(ValueError):
            make_material_operator(
                [{"density": {"kind": "table", "points": [[300.0, 1.0], [300.0, 2.0]]}}],
                np.zeros((2, 2, 2), dtype=np.int32),
                np.ones((2, 2, 2), dtype=bool),
                jax=jax,
                jnp=jnp,
            )


@unittest.skipUnless(_HAS_JAX, "the pinned WSL JAX environment is required")
class LesOperatorTests(unittest.TestCase):
    def test_shear_and_inactive_cells_match_cpu_solver(self) -> None:
        shape = (5, 4, 3)
        dx = 0.02
        coefficient = 0.17
        grid = np.indices(shape, dtype=float)
        velocity = np.stack(
            (
                0.35 * grid[1] + 0.12 * grid[2],
                -0.21 * grid[0] + 0.18 * grid[2],
                0.09 * grid[0] - 0.27 * grid[1],
            )
        )
        fluid = np.ones(shape, dtype=bool)
        fluid[0, 1, 1] = False
        fluid[2, 2, 1] = False
        fluid[-1, -1, -1] = False
        expected = _smagorinsky_eddy_viscosity(velocity, fluid, dx, coefficient)

        evaluate = make_les_operator(fluid, dx, coefficient, jax=jax, jnp=jnp)
        actual = np.asarray(evaluate(velocity))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
        self.assertTrue(np.all(actual[~fluid] == 0.0))
        self.assertGreater(float(np.max(actual[fluid])), 0.0)

    def test_les_operator_jits_and_zeroes_inactive_domain(self) -> None:
        shape = (4, 3, 2)
        fluid = np.ones(shape, dtype=bool)
        fluid[:, 0, :] = False
        velocity = np.zeros((3, *shape), dtype=float)
        velocity[0] = np.indices(shape, dtype=float)[1]
        evaluate = make_les_operator(fluid, 0.1, 0.2, jax=jax, jnp=jnp)
        actual = np.asarray(jax.jit(evaluate)(jnp.asarray(velocity)))
        self.assertTrue(np.all(actual[~fluid] == 0.0))
        self.assertTrue(np.all(actual[fluid] >= 0.0))
