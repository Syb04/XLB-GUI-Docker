import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workbench.results import field_specs, frames_result, slice_result
from workbench.schema import default_project


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        project = default_project()
        project["study"]["dt"] = 0.1
        project["materials"][0].update(
            density={"kind": "table", "points": [[300.0, 1000.0], [400.0, 1100.0]]},
            viscosity={"kind": "constant", "value": 0.002},
            heat_capacity={"kind": "constant", "value": 4000.0},
            conductivity={"kind": "constant", "value": 0.5},
        )
        (self.directory / "input.json").write_text(json.dumps(project), encoding="utf-8")
        shape = (5, 4, 3)
        fluid = np.ones(shape, dtype=bool)
        solid = np.zeros(shape, dtype=bool)
        solid[2, 2, 1] = True
        fluid[2, 2, 1] = False
        thermal = fluid | solid
        material_index = np.zeros(shape, dtype=np.int32)
        self.shape = shape
        self.fluid = fluid
        self.solid = solid
        (self.directory / "frames").mkdir()
        np.savez(
            self.directory / "frames" / "static.npz",
            fluid_mask=fluid,
            solid_mask=solid,
            thermal_mask=thermal,
            material_index=material_index,
            origin=[1.0, 2.0, 3.0],
            spacing=[0.2, 0.5, 1.0],
        )
        coordinates = np.indices(shape, dtype=float)
        temperature = 300.0 + 2.0 * coordinates[0] * 0.2 + 3.0 * coordinates[1] * 0.5
        velocity = np.zeros((3, *shape), dtype=float)
        velocity[0] = 2.0 * coordinates[1] * 0.5
        velocity[1] = 4.0 * coordinates[0] * 0.2
        for step in (2, 4):
            np.savez(
                self.directory / "frames" / f"step_{step:09d}.npz",
                velocity=velocity,
                pressure=np.full(shape, 5.0),
                temperature=temperature,
                eddy_viscosity=np.full(shape, 0.001),
            )
        # A file which is not in the atomic manifest is not publicly visible.
        np.savez(self.directory / "frames" / "step_000000003.npz", velocity=velocity)
        (self.directory / "frames.json").write_text(
            json.dumps({"frames": [{"step": 2, "time": 0.2, "file": "step_000000002.npz"},
                                   {"step": 4, "time": 0.4, "file": "step_000000004.npz"}]}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_manifest_and_public_field_contract(self):
        result = frames_result(self.directory)
        self.assertEqual(result["frames"], [{"step": 2, "time": 0.2}, {"step": 4, "time": 0.4}])
        ids = {field["id"] for field in result["fields"]}
        self.assertEqual(ids, {field["id"] for field in field_specs()})
        self.assertIn("velocity_x", ids)
        self.assertIn("vorticity", ids)
        self.assertIn("heat_flux", ids)

    def test_derived_fields_use_si_spacing_materials_and_masks(self):
        query = {"step": ["2"], "axis": ["z"], "index": ["0"]}
        density = slice_result(self.directory, {**query, "field": ["density"]})
        self.assertEqual(density["step"], 2)
        self.assertEqual(density["time"], 0.2)
        self.assertAlmostEqual(density["values"][0][0], 1000.0)
        velocity = slice_result(self.directory, {**query, "field": ["vorticity_z"]})
        self.assertAlmostEqual(velocity["values"][1][1], 2.0, places=10)
        heat_flux = slice_result(self.directory, {**query, "field": ["heat_flux_x"]})
        self.assertAlmostEqual(heat_flux["values"][1][1], -1.0, places=10)
        # The solid is hidden from flow fields while its thermal value remains
        # addressable through the thermal mask.
        solid_slice = slice_result(self.directory, {"step": ["2"], "axis": ["z"], "index": ["1"], "field": ["speed"]})
        self.assertIsNone(solid_slice["values"][2][2])
        temperature = slice_result(self.directory, {"step": ["2"], "axis": ["z"], "index": ["1"], "field": ["temperature"]})
        self.assertIsNotNone(temperature["values"][2][2])

    def test_unpublished_and_unsafe_steps_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            slice_result(self.directory, {"step": ["3"], "field": ["speed"], "axis": ["z"]})
        (self.directory / "frames.json").write_text(
            json.dumps({"frames": [{"step": 2, "time": 0.2, "file": "../outside.npz"}]}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            frames_result(self.directory)

    def test_derived_material_field_requires_input(self):
        (self.directory / "input.json").unlink()
        with self.assertRaises(ValueError):
            slice_result(self.directory, {"step": ["2"], "field": ["density"], "axis": ["z"]})

    def test_missing_solid_index_with_geometry_solids_is_rejected(self):
        project = json.loads((self.directory / "input.json").read_text(encoding="utf-8"))
        project["materials"].append(
            {
                "id": "solid-material",
                "density": {"kind": "constant", "value": 2700.0},
                "viscosity": {"kind": "constant", "value": 0.001},
                "heat_capacity": {"kind": "constant", "value": 900.0},
                "conductivity": {"kind": "constant", "value": 200.0},
            }
        )
        # Keep both declarations to ensure solid_material_id is not used as a
        # CAD-wide fallback when geometry.solids can contain multiple regions.
        project["geometry"]["solid_material_id"] = "solid-material"
        project["geometry"]["solids"] = [
            {
                "id": "insert",
                "name": "Insert",
                "origin": [1.0, 2.0, 3.0],
                "size": [0.2, 0.5, 1.0],
                "material_id": "solid-material",
            }
        ]
        (self.directory / "input.json").write_text(json.dumps(project), encoding="utf-8")
        material_index = np.zeros(self.shape, dtype=np.int32)
        material_index[self.solid] = -1
        np.savez(
            self.directory / "frames" / "static.npz",
            fluid_mask=self.fluid,
            solid_mask=self.solid,
            thermal_mask=self.fluid | self.solid,
            material_index=material_index,
            origin=[1.0, 2.0, 3.0],
            spacing=[0.2, 0.5, 1.0],
        )
        with self.assertRaisesRegex(ValueError, r"geometry\.solids"):
            slice_result(self.directory, {"step": ["2"], "field": ["density"], "axis": ["z"]})
        for field in ('temperature', 'speed'):
            result = slice_result(self.directory, {"step": ["2"], "field": [field], "axis": ["z"]})
            self.assertEqual(result['step'], 2)
            self.assertTrue(result['values'])


if __name__ == "__main__":
    unittest.main()
