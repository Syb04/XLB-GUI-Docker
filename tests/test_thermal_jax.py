from __future__ import annotations

import importlib.util
import unittest

import numpy as np


_HAS_JAX = importlib.util.find_spec("jax") is not None

if _HAS_JAX:
    import jax
    import jax.numpy as jnp

    from workbench.solver import _face_masks, _thermal_stability_numbers, _thermal_step
    from workbench.thermal_jax import make_thermal_operators


@unittest.skipUnless(_HAS_JAX, "the pinned JAX environment is required")
class ThermalJaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The thermal path can retain the full precision enabled by the
        # integrator while LBM distributions remain float32.
        try:
            jax.config.update("jax_enable_x64", True)
        except (AttributeError, RuntimeError, ValueError):
            pass

    def _operator(
        self,
        fluid: np.ndarray,
        thermal: np.ndarray,
        *,
        box: dict[str, dict[str, float | str]] | None = None,
        cad: dict[str, float | str] | None = None,
        links: list[tuple[np.ndarray, dict[str, float | str]]] | None = None,
        dx: float = 0.2,
        dt: float = 0.01,
    ):
        face_masks = _face_masks(tuple(fluid.shape), thermal)
        return make_thermal_operators(
            fluid,
            thermal,
            dx,
            dt,
            box or {},
            cad,
            face_masks,
            links or (),
            jax=jax,
            jnp=jnp,
        )

    def test_box_update_and_stability_match_numpy_reference_under_jit(self) -> None:
        shape = (5, 3, 2)
        rng = np.random.default_rng(8)
        fluid = np.ones(shape, dtype=bool)
        thermal = fluid.copy()
        velocity = rng.normal(size=(3, *shape)) * 0.04
        temperature = 300.0 + rng.normal(size=shape)
        density = 0.8 + rng.random(shape)
        heat_capacity = 0.9 + rng.random(shape)
        conductivity = 0.4 + rng.random(shape)
        box = {
            "xmin": {"type": "temperature", "value": 305.0},
            "xmax": {"type": "heat_flux", "value": 2.0},
            "ymin": {"type": "convection", "h": 3.0, "ambient_temperature": 290.0},
            "zmax": {"type": "temperature", "value": 310.0},
        }
        face_masks = _face_masks(shape, thermal)
        expected = _thermal_step(
            temperature,
            velocity,
            fluid,
            density,
            heat_capacity,
            conductivity,
            0.01,
            0.2,
            box,
            None,
            face_masks,
            [],
            thermal_mask=thermal,
        )
        expected_stability = _thermal_stability_numbers(
            density,
            heat_capacity,
            conductivity,
            fluid,
            0.01,
            0.2,
            box,
            None,
            face_masks,
            thermal_mask=thermal,
        )
        step, stability = self._operator(fluid, thermal, box=box)
        compiled_step = jax.jit(step)
        compiled_stability = jax.jit(stability)
        result = np.asarray(
            compiled_step(
                jnp.asarray(temperature),
                jnp.asarray(velocity),
                jnp.asarray(density),
                jnp.asarray(heat_capacity),
                jnp.asarray(conductivity),
            )
        )
        actual_stability = tuple(float(value) for value in compiled_stability(density, heat_capacity, conductivity))
        np.testing.assert_allclose(result, expected, rtol=2.0e-6, atol=2.0e-8)
        np.testing.assert_allclose(actual_stability, expected_stability, rtol=2.0e-6, atol=2.0e-8)

    def test_closed_box_heat_flux_conserves_energy_and_returns_device_scalars(self) -> None:
        shape = (4, 2, 2)
        fluid = np.ones(shape, dtype=bool)
        thermal = fluid.copy()
        velocity = np.zeros((3, *shape), dtype=float)
        temperature = np.full(shape, 300.0)
        density = np.ones(shape)
        heat_capacity = np.ones(shape)
        conductivity = np.ones(shape)
        box = {"xmin": {"type": "heat_flux", "value": 100.0}}
        step, stability = self._operator(fluid, thermal, box=box, dx=1.0, dt=0.001)
        compiled_step = jax.jit(step)
        compiled_stability = jax.jit(stability)
        field = jnp.asarray(temperature)
        for _ in range(10):
            field = compiled_step(field, velocity, density, heat_capacity, conductivity)
        # q*A*dt*N/(rho*cp*V) = 100*4*.001*10/(1*4*2*2).
        self.assertAlmostEqual(float(jnp.mean(field)), 300.25, places=8)
        values = compiled_stability(density, heat_capacity, conductivity)
        self.assertEqual(len(values), 3)
        self.assertTrue(all(getattr(value, "shape", None) == () for value in values))
        self.assertAlmostEqual(float(values[0]), 0.001, places=12)

    def test_cht_harmonic_conduction_and_local_cad_temperature_match_reference(self) -> None:
        shape = (8, 2, 1)
        fluid = np.zeros(shape, dtype=bool)
        fluid[:4, :, :] = True
        solid = np.zeros(shape, dtype=bool)
        solid[4:, :, :] = True
        thermal = fluid | solid
        velocity = np.zeros((3, *shape), dtype=float)
        temperature = np.full(shape, 300.0)
        density = np.where(solid, 2.0, 1.0)
        heat_capacity = np.where(solid, 3.0, 2.0)
        conductivity = np.where(solid, 5.0, 0.5)
        links = np.zeros((6, *shape), dtype=bool)
        links[0, 1, 0, 0] = True
        local = [(links, {"type": "temperature", "value": 320.0})]
        box = {"xmin": {"type": "temperature", "value": 400.0}}
        face_masks = _face_masks(shape, thermal)
        expected = _thermal_step(
            temperature,
            velocity,
            fluid,
            density,
            heat_capacity,
            conductivity,
            0.01,
            1.0,
            box,
            None,
            face_masks,
            [],
            thermal_mask=thermal,
            cad_link_actions=local,
        )
        step, stability = self._operator(fluid, thermal, box=box, links=local, dx=1.0, dt=0.01)
        actual = jax.jit(step)(temperature, velocity, density, heat_capacity, conductivity)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2.0e-6, atol=2.0e-8)
        expected_stability = _thermal_stability_numbers(
            density,
            heat_capacity,
            conductivity,
            fluid,
            0.01,
            1.0,
            box,
            None,
            face_masks,
            thermal_mask=thermal,
            cad_link_actions=local,
        )
        actual_stability = tuple(float(value) for value in jax.jit(stability)(density, heat_capacity, conductivity))
        np.testing.assert_allclose(actual_stability, expected_stability, rtol=2.0e-6, atol=2.0e-8)

    def test_legacy_cad_fallback_counts_multiple_wall_contacts(self) -> None:
        shape = (3, 3, 3)
        fluid = np.ones(shape, dtype=bool)
        fluid[1, 1, 1] = False
        thermal = fluid.copy()
        velocity = np.zeros((3, *shape), dtype=float)
        temperature = np.full(shape, 300.0)
        density = np.ones(shape)
        heat_capacity = np.ones(shape)
        conductivity = np.ones(shape)
        cad = {"type": "heat_flux", "value": 10.0}
        face_masks = _face_masks(shape, thermal)
        expected = _thermal_step(
            temperature,
            velocity,
            fluid,
            density,
            heat_capacity,
            conductivity,
            0.01,
            1.0,
            {},
            cad,
            face_masks,
            [],
            thermal_mask=thermal,
        )
        step, stability = self._operator(fluid, thermal, cad=cad, dx=1.0, dt=0.01)
        actual = jax.jit(step)(temperature, velocity, density, heat_capacity, conductivity)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2.0e-6, atol=2.0e-8)
        # The centre cell is not fluid, so all six surrounding cells receive a
        # single contact.  It also ensures the fallback remains a wall source
        # rather than a wrapped periodic interaction.
        self.assertAlmostEqual(float(np.asarray(actual)[1, 1, 0]), 300.1, places=10)
        expected_stability = _thermal_stability_numbers(
            density,
            heat_capacity,
            conductivity,
            fluid,
            0.01,
            1.0,
            {},
            cad,
            face_masks,
            thermal_mask=thermal,
        )
        actual_stability = tuple(float(value) for value in jax.jit(stability)(density, heat_capacity, conductivity))
        np.testing.assert_allclose(actual_stability, expected_stability, rtol=2.0e-6, atol=2.0e-8)

    def test_legacy_cad_fallback_stability_counts_all_six_contacts(self) -> None:
        shape = (3, 3, 3)
        fluid = np.zeros(shape, dtype=bool)
        fluid[1, 1, 1] = True
        thermal = fluid.copy()
        velocity = np.zeros((3, *shape), dtype=float)
        temperature = np.full(shape, 300.0)
        density = np.ones(shape)
        heat_capacity = np.ones(shape)
        conductivity = np.ones(shape)
        cad = {"type": "temperature", "value": 320.0}
        face_masks = _face_masks(shape, thermal)
        expected = _thermal_step(
            temperature,
            velocity,
            fluid,
            density,
            heat_capacity,
            conductivity,
            0.01,
            1.0,
            {},
            cad,
            face_masks,
            [],
            thermal_mask=thermal,
        )
        step, stability = self._operator(fluid, thermal, cad=cad, dx=1.0, dt=0.01)
        actual = jax.jit(step)(temperature, velocity, density, heat_capacity, conductivity)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2.0e-6, atol=2.0e-8)
        self.assertAlmostEqual(float(np.asarray(actual)[1, 1, 1]), 302.4, places=10)
        expected_stability = _thermal_stability_numbers(
            density,
            heat_capacity,
            conductivity,
            fluid,
            0.01,
            1.0,
            {},
            cad,
            face_masks,
            thermal_mask=thermal,
        )
        actual_stability = tuple(float(value) for value in jax.jit(stability)(density, heat_capacity, conductivity))
        np.testing.assert_allclose(actual_stability, expected_stability, rtol=2.0e-6, atol=2.0e-8)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
