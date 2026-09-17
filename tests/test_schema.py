import copy
import unittest
import warnings

import numpy as np

try:
    from workbench.schema import PropertyClampWarning, default_project, evaluate_property, normalise_gravity, validate_project
except ModuleNotFoundError:  # Running from the repository root.
    from xlb_workbench.workbench.schema import PropertyClampWarning, default_project, evaluate_property, normalise_gravity, validate_project


class SchemaTests(unittest.TestCase):
    def test_monitor_validation_and_restart_compatibility(self):
        from workbench.restart import project_fingerprint
        project = default_project()
        baseline = project_fingerprint(project)
        project['study']['monitor'] = {'tolerance': 0.001, 'consecutive_samples': 3}
        self.assertEqual(validate_project(project)['study']['monitor'], project['study']['monitor'])
        self.assertEqual(project_fingerprint(project), baseline)
        for descriptor in [None, [], {'tolerance': 0}, {'tolerance': float('nan')}, {'tolerance': True},
                           {'consecutive_samples': 0}, {'consecutive_samples': 101},
                           {'consecutive_samples': 1.5}, {'consecutive_samples': True}, {'auto_stop': True}]:
            project['study']['monitor'] = descriptor
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):
                validate_project(project)
        del project['study']['monitor']
        self.assertNotIn('monitor', validate_project(project)['study'])
        self.assertEqual(project_fingerprint(project), baseline)

    def test_ram_device_roundtrip_and_batch_validation(self):
        project = default_project()
        project["study"]["device"] = "cuda:0-ram"
        self.assertEqual(validate_project(project)["study"]["gpu_batch_cells"], 65536)
        for size in (1024, 1048576):
            project["study"]["gpu_batch_cells"] = size
            self.assertEqual(validate_project(project)["study"]["gpu_batch_cells"], size)
        for size in (True, 1023, 1048577, 2048.5):
            project["study"]["gpu_batch_cells"] = size
            with self.assertRaisesRegex(ValueError, "gpu_batch_cells"):
                validate_project(project)

    def test_default_is_valid_and_uses_stable_si_scaling(self):
        project = default_project()
        validated = validate_project(project)
        self.assertEqual(validated["schema_version"], 1)
        self.assertEqual(validated["mesh"]["cells"], [30, 10, 10])
        self.assertEqual(validated["study"]["dt"], 0.01)
        self.assertEqual(validated["geometry"]["size"], [0.06, 0.02, 0.02])
        self.assertEqual(validated["geometry"]["size"][0] / validated["mesh"]["cells"][0], 0.002)
        self.assertEqual(validated["geometry"]["size"][1] / validated["mesh"]["cells"][1], 0.002)
        self.assertEqual(validated["boundaries"][-1]["thermal"], {"type": "heat_flux", "value": 1000.0})

    def test_validation_does_not_mutate_input(self):
        project = default_project()
        original = copy.deepcopy(project)
        validate_project(project)
        self.assertEqual(project, original)

    def test_property_constant_and_table_scalar_and_array(self):
        constant = {"kind": "constant", "value": 2.5}
        self.assertEqual(evaluate_property(constant, 300.0), 2.5)
        table = {"kind": "table", "points": [[273.15, 1.0], [373.15, 3.0]]}
        self.assertAlmostEqual(evaluate_property(table, 323.15), 2.0)
        values = evaluate_property(table, np.asarray([273.15, 323.15, 373.15]))
        np.testing.assert_allclose(values, [1.0, 2.0, 3.0])

    def test_property_table_clamps_with_explicit_warning(self):
        table = {"kind": "table", "points": [[1.0, 1.0], [9.0, 3.0]]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = evaluate_property(table, np.asarray([0.5, 5.0, 20.0]))
        np.testing.assert_allclose(values, [1.0, 2.0, 3.0])
        self.assertTrue(any(issubclass(item.category, PropertyClampWarning) for item in caught))
        with self.assertRaises(ValueError):
            evaluate_property(table, 0.0)

    def test_invalid_property_tables_are_rejected(self):
        base = default_project()
        for points in (
            [[300.0, 1.0]],
            [[300.0, 1.0], [300.0, 2.0]],
            [[301.0, 1.0], [300.0, 2.0]],
            [[300.0, float("nan")], [301.0, 2.0]],
        ):
            project = copy.deepcopy(base)
            project["materials"][0]["viscosity"] = {"kind": "table", "points": points}
            with self.assertRaises(ValueError):
                validate_project(project)

    def test_invalid_ids_refs_booleans_and_duplicate_faces(self):
        project = default_project()
        project["physics"]["flow"] = 1
        with self.assertRaises(ValueError):
            validate_project(project)

        project = default_project()
        project["physics"]["material_id"] = "missing"
        with self.assertRaises(ValueError):
            validate_project(project)

        project = default_project()
        project["boundaries"][1]["face"] = "xmin"
        with self.assertRaises(ValueError):
            validate_project(project)

        project = default_project()
        project["boundaries"][1]["id"] = project["boundaries"][0]["id"]
        with self.assertRaises(ValueError):
            validate_project(project)

    def test_finite_and_isotropic_geometry_and_bounded_memory(self):
        project = default_project()
        project["geometry"]["size"][0] = float("inf")
        with self.assertRaises(ValueError):
            validate_project(project)

        project = default_project()
        project["geometry"]["size"] = [0.061, 0.02, 0.02]
        with self.assertRaises(ValueError):
            validate_project(project)

        project = default_project()
        project["mesh"]["cells"] = [512, 512, 8]
        with self.assertRaises(ValueError):
            validate_project(project)

    def test_wall_and_cad_boundary_conventions(self):
        project = default_project()
        project["boundaries"][-1] = {
            "id": "cad-wall",
            "name": "CAD wall",
            "face": "cad",
            "flow": {"type": "wall"},
            "thermal": {"type": "convection", "h": 10.0, "ambient_temperature": 293.15},
        }
        project["geometry"] = {"kind": "cad", "asset_id": "a" * 32, "role": "fluid"}
        self.assertEqual(validate_project(project)["boundaries"][-1]["flow"], {"type": "wall"})
        project["boundaries"][-1]["flow"] = {"type": "pressure", "value": 0.0}
        with self.assertRaises(ValueError):
            validate_project(project)

    def test_turbulence_descriptor_defaults_and_validation(self):
        project = default_project()
        project["physics"].pop("turbulence")
        normalized = validate_project(project)
        self.assertEqual(normalized["physics"]["turbulence"]["model"], "laminar")
        project["physics"]["turbulence"] = {
            "model": "smagorinsky",
            "smagorinsky_constant": 0.2,
            "turbulent_prandtl": 0.85,
        }
        normalized = validate_project(project)
        self.assertEqual(normalized["physics"]["turbulence"]["model"], "smagorinsky")
        self.assertEqual(normalized["physics"]["turbulence"]["smagorinsky_constant"], 0.2)
        for invalid in (
            {"model": "k-epsilon"},
            {"model": "smagorinsky", "smagorinsky_constant": 0.0},
            {"model": "smagorinsky", "turbulent_prandtl": float("nan")},
            {"model": "smagorinsky", "unexpected": 1.0},
        ):
            project["physics"]["turbulence"] = invalid
            with self.assertRaises(ValueError):
                validate_project(project)

    def test_optional_gravity_defaults_roundtrip_and_legacy_omission(self):
        defaults = {
            "enabled": False,
            "mode": "uniform",
            "vector": [0.0, 0.0, -9.80665],
            "reference_temperature": 293.15,
        }
        self.assertEqual(normalise_gravity(None), defaults)
        self.assertEqual(normalise_gravity({}), defaults)
        self.assertEqual(validate_project(default_project())["physics"]["gravity"], defaults)

        legacy = default_project()
        legacy["physics"].pop("gravity")
        normalized = validate_project(legacy)
        self.assertNotIn("gravity", normalized["physics"])

        project = default_project()
        project["physics"]["gravity"] = {
            "enabled": True,
            "mode": "buoyancy",
            "vector": [1, -2, -9.81],
            "reference_temperature": 310,
        }
        project["physics"]["flow"] = False
        normalized = validate_project(project)
        self.assertEqual(normalized["physics"]["gravity"], {
            "enabled": True,
            "mode": "buoyancy",
            "vector": [1.0, -2.0, -9.81],
            "reference_temperature": 310.0,
        })

    def test_gravity_rejects_unknown_and_invalid_values(self):
        base = default_project()
        invalid_values = (
            {"unexpected": 1},
            {"enabled": 1},
            {"mode": "ambiguous"},
            {"vector": [0.0, 0.0]},
            {"vector": [0.0, 0.0, float("nan")]},
            {"reference_temperature": 0.0},
            [],
            "uniform",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalise_gravity(value)
                project = copy.deepcopy(base)
                project["physics"]["gravity"] = value
                with self.assertRaises(ValueError):
                    validate_project(project)

    def test_cht_solid_regions_and_material_references(self):
        project = default_project()
        solid = copy.deepcopy(project["materials"][0])
        solid.update({"id": "aluminum", "name": "Aluminum"})
        project["materials"].append(solid)
        project["geometry"]["solids"] = [
            {
                "id": "insert",
                "name": "Conductive insert",
                "origin": [0.02, 0.004, 0.004],
                "size": [0.01, 0.012, 0.012],
                "material_id": "aluminum",
            }
        ]
        normalized = validate_project(project)
        self.assertEqual(normalized["geometry"]["solids"][0]["material_id"], "aluminum")
        project["geometry"]["solids"][0]["material_id"] = "missing"
        with self.assertRaises(ValueError):
            validate_project(project)
        project = default_project()
        project["geometry"]["solids"] = [
            {
                "id": "outside",
                "name": "Outside",
                "origin": [0.059, 0.0, 0.0],
                "size": [0.002, 0.002, 0.002],
                "material_id": "water",
            }
        ]
        with self.assertRaisesRegex(ValueError, "inside"):
            validate_project(project)

    def test_individual_cad_patch_selectors_allow_flow_conditions(self):
        project = default_project()
        project["geometry"] = {"kind": "cad", "asset_id": "a" * 32, "role": "fluid"}
        project["boundaries"][-1] = {
            "id": "cad-inlet",
            "name": "CAD inlet",
            "face": "cad:surface-12",
            "flow": {"type": "velocity", "velocity": [0.01, 0.0, 0.0]},
            "thermal": {"type": "temperature", "value": 293.15},
        }
        self.assertEqual(validate_project(project)["boundaries"][-1]["face"], "cad:surface-12")
        project["boundaries"][-1]["face"] = "cad:"
        with self.assertRaises(ValueError):
            validate_project(project)
        project["boundaries"][-1]["face"] = "cad:surface-12"
        project["boundaries"][-1]["flow"] = {"type": "bad"}
        with self.assertRaises(ValueError):
            validate_project(project)


if __name__ == "__main__":
    unittest.main()
