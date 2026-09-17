"""Meaningful CPU checks for the real XLB workbench solver."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


_HAS_XLB = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("xlb") is not None
if _HAS_XLB:
    from workbench.solver import simulate
from workbench.solver import _cad_boundary_actions, _face_masks, _smagorinsky_eddy_viscosity, _thermal_step


def _project(*, flow: bool = True, thermal: bool = True, steps: int = 6) -> dict:
    return {
        "schema_version": 1,
        "name": "solver test",
        "geometry": {"kind": "box", "size": [0.04, 0.02, 0.02], "role": "fluid"},
        "materials": [
            {
                "id": "water",
                "name": "Water",
                "density": {"kind": "constant", "value": 998.0},
                "viscosity": {"kind": "table", "points": [[280.0, 0.0008], [320.0, 0.0016]]},
                "heat_capacity": {"kind": "constant", "value": 4182.0},
                "conductivity": {"kind": "constant", "value": 0.6},
            }
        ],
        "physics": {"flow": flow, "thermal": thermal, "material_id": "water", "initial_temperature": 300.0},
        "boundaries": [
            {"id": "inlet", "name": "Inlet", "face": "xmin", "flow": {"type": "velocity", "velocity": [0.01, 0.0, 0.0]}, "thermal": {"type": "temperature", "value": 305.0}},
            {"id": "outlet", "name": "Outlet", "face": "xmax", "flow": {"type": "pressure", "value": 0.0}, "thermal": {"type": "adiabatic"}},
        ],
        "mesh": {"cells": [8, 4, 4]},
        "study": {"steps": steps, "output_interval": 3, "dt": 0.001, "device": "cpu"},
    }


def _mesh(shape: tuple[int, int, int] = (8, 4, 4), spacing: float = 0.005) -> dict:
    return {
        "fluid_mask": np.ones(shape, dtype=bool),
        "origin": np.array([0.0, 0.0, 0.0]),
        "spacing": np.array([spacing, spacing, spacing]),
        "shape": list(shape),
    }


@unittest.skipUnless(_HAS_XLB, "the pinned WSL XLB/JAX environment is required")
class SolverTests(unittest.TestCase):
    def test_cpu_run_uses_xlb_and_writes_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xlb-solver-") as directory:
            run_dir = Path(directory)
            progress: list[dict] = []
            result = simulate(_project(), _mesh(), run_dir, on_progress=progress.append)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["steps"], 6)
            self.assertEqual(result["diagnostics"]["velocity_set"], "D3Q27")
            self.assertEqual(result["diagnostics"]["collision"], "BGK")
            self.assertTrue(progress)
            self.assertTrue({"step", "progress", "residual", "temperature_min", "temperature_max"} <= progress[-1].keys())

            with np.load(run_dir / "fields.npz") as fields:
                self.assertEqual(fields["velocity"].shape, (3, 8, 4, 4))
                self.assertEqual(fields["pressure"].shape, (8, 4, 4))
                self.assertEqual(fields["temperature"].shape, (8, 4, 4))
                self.assertEqual(fields["fluid_mask"].dtype, np.bool_)
                self.assertTrue(np.isfinite(fields["velocity"]).all())
                self.assertTrue(np.isfinite(fields["pressure"]).all())
                self.assertTrue(np.isfinite(fields["temperature"]).all())
                # The inlet equilibrium is a computed non-zero state; this
                # catches a demo/fake field implementation that only writes zeros.
                self.assertGreater(float(np.max(np.abs(fields["velocity"]))), 0.0)
                self.assertGreater(float(fields["temperature"][0, 1, 1]), 300.0)
                self.assertLess(float(fields["temperature"][0, 1, 1]), 305.0)

            with (run_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["step"]) for row in rows], [0, 3, 6])
            self.assertIn("STRUCTURED_POINTS", (run_dir / "fields.vtk").read_text(encoding="utf-8"))

    def test_flow_disabled_temperature_dependent_properties_and_wall_temperature(self) -> None:
        project = _project(flow=False, thermal=True, steps=4)
        project["boundaries"] = [
            {"id": "hot", "face": "xmin", "flow": {"type": "no-slip"}, "thermal": {"type": "temperature", "value": 310.0}},
            {"id": "cold", "face": "xmax", "flow": {"type": "no-slip"}, "thermal": {"type": "temperature", "value": 290.0}},
        ]
        with tempfile.TemporaryDirectory(prefix="xlb-thermal-") as directory:
            result = simulate(project, _mesh(), Path(directory))
            self.assertEqual(result["status"], "completed")
            self.assertGreater(result["diagnostics"]["tau_max"], result["diagnostics"]["tau_min"])
            with np.load(Path(directory) / "fields.npz") as fields:
                temperature = fields["temperature"]
                self.assertGreater(float(np.mean(temperature[0, :, :])), 300.0)
                self.assertLess(float(np.mean(temperature[-1, :, :])), 300.0)
                self.assertGreaterEqual(float(np.min(temperature)), 290.0)
                self.assertLessEqual(float(np.max(temperature)), 310.0)

    def test_unstable_tau_fails_before_running(self) -> None:
        project = _project(steps=1)
        project["study"]["dt"] = 1.0e-6
        with tempfile.TemporaryDirectory(prefix="xlb-invalid-") as directory:
            with self.assertRaisesRegex(ValueError, "relaxation time|tau"):
                simulate(project, _mesh(), Path(directory))


class ThermalFiniteVolumeTests(unittest.TestCase):
    def _arrays(self, shape: tuple[int, int, int], value: float = 1.0) -> tuple[np.ndarray, ...]:
        fluid = np.ones(shape, dtype=bool)
        return (
            fluid,
            np.zeros((3, *shape), dtype=float),
            np.full(shape, value, dtype=float),
            np.full(shape, value, dtype=float),
            np.full(shape, value, dtype=float),
        )

    def test_steady_1d_dirichlet_conduction_matches_cell_centres(self) -> None:
        shape = (20, 2, 2)
        fluid, velocity, density, heat_capacity, conductivity = self._arrays(shape)
        temperature = np.full(shape, 273.5)
        face_masks = _face_masks(shape, fluid)
        box = {
            "xmin": {"type": "temperature", "value": 274.0},
            "xmax": {"type": "temperature", "value": 273.0},
        }
        for _ in range(7000):
            temperature = _thermal_step(
                temperature, velocity, fluid, density, heat_capacity, conductivity,
                dt=0.01, dx=1.0, box_thermal=box, cad_thermal=None,
                face_masks=face_masks, warnings=[],
            )
        expected = 274.0 - (np.arange(shape[0]) + 0.5) / shape[0]
        np.testing.assert_allclose(temperature[:, 0, 0], expected, atol=1.0e-3)

    def test_closed_volume_heat_flux_conserves_energy_and_cad_flux_is_internal(self) -> None:
        shape = (4, 2, 2)
        fluid, velocity, density, heat_capacity, conductivity = self._arrays(shape, value=1.0)
        face_masks = _face_masks(shape, fluid)
        temperature = np.full(shape, 300.0)
        box = {"xmin": {"type": "heat_flux", "value": 100.0}}
        for _ in range(10):
            temperature = _thermal_step(
                temperature, velocity, fluid, density, heat_capacity, conductivity,
                dt=0.001, dx=1.0, box_thermal=box, cad_thermal=None,
                face_masks=face_masks, warnings=[],
            )
        # q*A*dt*N / (rho*cp*V) = 100*4*.001*10/(1*4*2*2).
        self.assertAlmostEqual(float(np.mean(temperature)), 300.25, places=8)

        # A CAD heat flux has no effect on an all-fluid box: only internal
        # fluid/solid contacts are CAD wall contacts.
        untouched = np.full(shape, 300.0)
        for _ in range(3):
            untouched = _thermal_step(
                untouched, velocity, fluid, density, heat_capacity, conductivity,
                dt=0.001, dx=1.0, box_thermal={},
                cad_thermal={"type": "heat_flux", "value": 100.0},
                face_masks=face_masks, warnings=[],
            )
        np.testing.assert_allclose(untouched, 300.0)

        obstacle_shape = (5, 2, 2)
        obstacle_fluid, obstacle_velocity, obstacle_density, obstacle_cp, obstacle_k = self._arrays(obstacle_shape, value=1.0)
        obstacle_fluid[2, :, :] = False
        obstacle_masks = _face_masks(obstacle_shape, obstacle_fluid)
        obstacle_temperature = np.full(obstacle_shape, 300.0)
        obstacle_temperature = _thermal_step(
            obstacle_temperature, obstacle_velocity, obstacle_fluid, obstacle_density, obstacle_cp, obstacle_k,
            dt=0.001, dx=1.0, box_thermal={},
            cad_thermal={"type": "heat_flux", "value": 100.0},
            face_masks=obstacle_masks, warnings=[],
        )
        self.assertTrue(np.all(obstacle_temperature[0, :, :] == 300.0))
        self.assertTrue(np.all(obstacle_temperature[-1, :, :] == 300.0))
        self.assertTrue(np.all(obstacle_temperature[1, :, :] > 300.0))
        self.assertTrue(np.all(obstacle_temperature[3, :, :] > 300.0))

    def test_composite_fluid_solid_slab_matches_harmonic_resistance(self) -> None:
        """Fluid-solid conduction uses the exact harmonic interface flux."""

        shape = (24, 1, 1)
        fluid = np.zeros(shape, dtype=bool)
        fluid[:12, :, :] = True
        solid = np.zeros(shape, dtype=bool)
        solid[12:, :, :] = True
        thermal = fluid | solid
        velocity = np.zeros((3, *shape), dtype=float)
        density = np.ones(shape, dtype=float)
        heat_capacity = np.ones(shape, dtype=float)
        conductivity = np.where(solid, 4.0, 1.0)
        temperature = np.full(shape, 300.0)
        face_masks = _face_masks(shape, thermal)
        box = {
            "xmin": {"type": "temperature", "value": 400.0},
            "xmax": {"type": "temperature", "value": 300.0},
        }
        for _ in range(5000):
            temperature = _thermal_step(
                temperature,
                velocity,
                fluid,
                density,
                heat_capacity,
                conductivity,
                dt=0.05,
                dx=1.0,
                box_thermal=box,
                cad_thermal=None,
                face_masks=face_masks,
                warnings=[],
                thermal_mask=thermal,
            )
        # The two half-cell boundaries and the fluid-solid harmonic face give
        # R = 12/k_fluid + 12/k_solid = 15 in these cell units.
        heat_rate = (400.0 - 300.0) / 15.0
        expected = np.empty(24, dtype=float)
        expected[:12] = 400.0 - heat_rate * (np.arange(12) + 0.5)
        expected[12:] = 400.0 - heat_rate * (12.0 + (np.arange(12) + 0.5) / 4.0)
        np.testing.assert_allclose(temperature[:, 0, 0], expected, atol=2.0e-2)

    def test_composite_fluid_solid_conduction_conserves_weighted_energy(self) -> None:
        shape = (8, 2, 1)
        fluid = np.zeros(shape, dtype=bool)
        fluid[:4, :, :] = True
        solid = np.zeros(shape, dtype=bool)
        solid[4:, :, :] = True
        thermal = fluid | solid
        velocity = np.zeros((3, *shape), dtype=float)
        density = np.where(solid, 2.0, 1.0)
        heat_capacity = np.where(solid, 3.0, 2.0)
        conductivity = np.where(solid, 5.0, 0.5)
        temperature = np.where(fluid, 400.0, 300.0)
        face_masks = _face_masks(shape, thermal)
        energy_before = float(np.sum(density * heat_capacity * temperature))
        for _ in range(20):
            temperature = _thermal_step(
                temperature,
                velocity,
                fluid,
                density,
                heat_capacity,
                conductivity,
                dt=0.01,
                dx=1.0,
                box_thermal={},
                cad_thermal=None,
                face_masks=face_masks,
                warnings=[],
                thermal_mask=thermal,
            )
        energy_after = float(np.sum(density * heat_capacity * temperature))
        self.assertAlmostEqual(energy_after, energy_before, places=10)

    def test_cad_temperature_port_sets_incoming_advective_ghost(self) -> None:
        shape = (6, 1, 1)
        fluid = np.ones(shape, dtype=bool)
        velocity = np.zeros((3, *shape), dtype=float)
        velocity[0] = 1.0
        density = np.ones(shape, dtype=float)
        heat_capacity = np.ones(shape, dtype=float)
        # Make the conductive wall contribution negligible so this isolates
        # the prescribed incoming characteristic at the x- CAD port.
        conductivity = np.full(shape, 1.0e-8)
        temperature = np.full(shape, 300.0)
        links = np.zeros((6, *shape), dtype=bool)
        links[0, 2, 0, 0] = True
        result = _thermal_step(
            temperature,
            velocity,
            fluid,
            density,
            heat_capacity,
            conductivity,
            dt=0.1,
            dx=1.0,
            box_thermal={},
            cad_thermal=None,
            face_masks=_face_masks(shape, fluid),
            warnings=[],
            cad_link_actions=[(links, {"type": "temperature", "value": 310.0})],
        )
        self.assertGreater(float(result[2, 0, 0]), 300.9)
        self.assertLess(float(result[3, 0, 0]), 300.01)


class TurbulenceTests(unittest.TestCase):
    def test_smagorinsky_eddy_viscosity_is_zero_for_zero_strain_and_nonnegative(self) -> None:
        shape = (7, 5, 3)
        fluid = np.ones(shape, dtype=bool)
        fluid[0, :, :] = False
        velocity = np.zeros((3, *shape), dtype=float)
        self.assertEqual(float(np.max(_smagorinsky_eddy_viscosity(velocity, fluid, 0.1, 0.17))), 0.0)

        # u_x = y is a constant shear with a strictly positive invariant.
        velocity[0] = np.arange(shape[1], dtype=float)[None, :, None]
        eddy = _smagorinsky_eddy_viscosity(velocity, fluid, 0.1, 0.17)
        self.assertTrue(np.isfinite(eddy).all())
        self.assertTrue(np.all(eddy >= 0.0))
        self.assertGreater(float(np.max(eddy)), 0.0)
        self.assertTrue(np.all(eddy[~fluid] == 0.0))


class CADLinkResolutionTests(unittest.TestCase):
    def test_overlapping_or_unmapped_selector_links_fail_explicitly(self) -> None:
        shape = (4, 2, 1)
        fluid = np.ones(shape, dtype=bool)
        thermal = fluid.copy()
        links = np.zeros((6, *shape), dtype=bool)
        links[0, 1, :, :] = True
        with self.assertRaisesRegex(ValueError, "overlap"):
            _cad_boundary_actions(
                {"cad:a": links, "cad:b": links.copy()},
                {},
                None,
                {
                    "cad:a": {"flow": {"type": "wall"}, "thermal": {"type": "adiabatic"}},
                    "cad:b": {"flow": {"type": "wall"}, "thermal": {"type": "adiabatic"}},
                },
                fluid,
                thermal,
            )
        with self.assertRaisesRegex(ValueError, "no mapped"):
            _cad_boundary_actions(
                {},
                {},
                None,
                {"cad:missing": {"flow": {"type": "wall"}, "thermal": {"type": "adiabatic"}}},
                fluid,
                thermal,
            )

    def test_nonadiabatic_cad_link_on_conductive_solid_is_rejected(self) -> None:
        shape = (4, 1, 1)
        fluid = np.zeros(shape, dtype=bool)
        fluid[:2] = True
        solid = np.zeros(shape, dtype=bool)
        solid[2:] = True
        thermal = fluid | solid
        links = np.zeros((6, *shape), dtype=bool)
        links[1, 1, 0, 0] = True  # x+ from fluid into the solid
        with self.assertRaisesRegex(ValueError, "fluid-solid"):
            _cad_boundary_actions(
                {"cad:interface": links},
                {},
                None,
                {"cad:interface": {"flow": {"type": "wall"}, "thermal": {"type": "temperature", "value": 320.0}}},
                fluid,
                thermal,
            )


@unittest.skipUnless(_HAS_XLB, "the pinned WSL XLB/JAX environment is required")
class CADPatchTests(unittest.TestCase):
    def test_selector_cad_velocity_and_temperature_links_change_real_fields(self) -> None:
        project = _project(steps=4)
        project["boundaries"] = [
            {
                "id": "port",
                "name": "CAD port",
                "face": "cad:port",
                "flow": {"type": "velocity", "velocity": [0.01, 0.0, 0.0]},
                "thermal": {"type": "temperature", "value": 310.0},
            }
        ]
        mesh = _mesh()
        links = np.zeros((6, *mesh["fluid_mask"].shape), dtype=bool)
        links[0, 2, :, :] = True  # x- normal, incoming for positive u_x
        mesh["boundary_links"] = {"cad:port": links}
        with tempfile.TemporaryDirectory(prefix="xlb-cad-port-") as directory:
            simulate(project, mesh, Path(directory))
            with np.load(Path(directory) / "fields.npz") as fields:
                self.assertGreater(float(np.max(np.abs(fields["velocity"][:, 2, :, :]))), 0.0)
                self.assertGreater(float(np.mean(fields["temperature"][2, :, :])), 300.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
