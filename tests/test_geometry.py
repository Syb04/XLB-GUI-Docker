import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover - exercised in minimal installations
    trimesh = None

try:
    import gmsh
except Exception:  # pragma: no cover - optional native dependency
    gmsh = None

try:
    from workbench.geometry import build_mesh, import_cad
    from workbench.schema import default_project
except ModuleNotFoundError:  # Running from the repository root.
    from xlb_workbench.workbench.geometry import build_mesh, import_cad
    from xlb_workbench.workbench.schema import default_project


@unittest.skipIf(trimesh is None, "trimesh is required for CAD import tests")
class GeometryTests(unittest.TestCase):
    def test_box_mesh_isotropic_and_fluid(self):
        project = default_project()
        with tempfile.TemporaryDirectory() as temp:
            mesh = build_mesh(project, temp)
        self.assertEqual(mesh["shape"], [30, 10, 10])
        self.assertEqual(mesh["fluid_mask"].dtype, np.bool_)
        self.assertEqual(mesh["fluid_mask"].shape, (30, 10, 10))
        self.assertTrue(mesh["fluid_mask"].all())
        self.assertFalse(mesh["solid_mask"].any())
        self.assertTrue(mesh["thermal_mask"].all())
        self.assertTrue((mesh["material_index"] == 0).all())
        self.assertEqual(mesh["boundary_links"]["xmin"].shape, (6, 30, 10, 10))
        self.assertEqual(int(mesh["boundary_links"]["xmin"].sum()), 100)
        np.testing.assert_allclose(mesh["origin"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(mesh["spacing"], [0.002, 0.002, 0.002])

    def test_box_fluid_cht_region_assigns_solid_material(self):
        project = default_project()
        solid = dict(project["materials"][0])
        solid.update({"id": "copper", "name": "Copper"})
        project["materials"].append(solid)
        project["geometry"]["solids"] = [
            {
                "id": "heater",
                "name": "Embedded heater",
                "origin": [0.02, 0.004, 0.004],
                "size": [0.01, 0.012, 0.012],
                "material_id": "copper",
            }
        ]
        with tempfile.TemporaryDirectory() as temp:
            mesh = build_mesh(project, temp)
        self.assertGreater(int(mesh["solid_mask"].sum()), 0)
        self.assertLess(int(mesh["fluid_mask"].sum()), 30 * 10 * 10)
        self.assertTrue(np.all(mesh["material_index"][mesh["solid_mask"]] == 1))
        self.assertTrue(np.all(mesh["thermal_mask"][mesh["solid_mask"]]))

    def test_box_obstacle_masks_solid_centres(self):
        project = default_project()
        project["geometry"] = {
            "kind": "box",
            "size": [0.01, 0.01, 0.01],
            "origin": [-0.005, -0.005, -0.005],
            "asset_id": None,
            "role": "obstacle",
            "computational_box": {"size": [0.02, 0.02, 0.02], "origin": [-0.01, -0.01, -0.01]},
        }
        project["mesh"]["cells"] = [20, 20, 20]
        with tempfile.TemporaryDirectory() as temp:
            mesh = build_mesh(project, temp)
        self.assertEqual(mesh["shape"], [20, 20, 20])
        self.assertEqual(int(mesh["fluid_mask"].sum()), 7000)
        self.assertFalse(mesh["fluid_mask"][9:11, 9:11, 9:11].all())

    def test_import_cube_mm_persists_source_and_si_surface(self):
        cube = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "cube.stl"
            cube.export(source)
            metadata = import_cad(source, root / "assets", unit="mm")
            self.assertEqual(metadata["original_name"], "cube.stl")
            self.assertEqual(metadata["source_extension"], ".stl")
            self.assertEqual(metadata["unit"], "mm")
            self.assertEqual(metadata["bounds"], [[-0.005, -0.005, -0.005], [0.005, 0.005, 0.005]])
            asset = root / "assets" / metadata["id"]
            self.assertTrue((asset / "metadata.json").is_file())
            self.assertTrue((asset / "surface.npz").is_file())
            self.assertTrue((asset / "source.stl").is_file())
            self.assertEqual((asset / "source.stl").read_bytes(), source.read_bytes())
            on_disk = json.loads((asset / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["id"], metadata["id"])
            self.assertEqual(len(metadata["surface_groups"]), 6)
            self.assertEqual(len(metadata["triangle_groups"]), metadata["triangle_count"])
            self.assertEqual(sum(group["triangle_count"] for group in metadata["surface_groups"]), metadata["triangle_count"])
            with np.load(asset / "surface.npz") as surface:
                np.testing.assert_allclose(surface["vertices"].min(axis=0), [-0.005] * 3)
                np.testing.assert_allclose(surface["vertices"].max(axis=0), [0.005] * 3)

            project = default_project()
            project["geometry"] = {"kind": "cad", "asset_id": metadata["id"], "role": "fluid"}
            project["mesh"]["cells"] = [10, 10, 10]
            mesh = build_mesh(project, root / "assets")
            self.assertEqual(mesh["shape"], [10, 10, 10])
            self.assertTrue(mesh["fluid_mask"].all())

    def test_import_removes_exact_duplicate_triangles_without_changing_surface(self):
        cube = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "duplicate-face.obj"
            lines = [f"v {x} {y} {z}" for x, y, z in cube.vertices]
            lines.extend("f " + " ".join(str(int(index) + 1) for index in face) for face in cube.faces)
            lines.append("f " + " ".join(str(int(index) + 1) for index in cube.faces[0]))
            source.write_text("\n".join(lines) + "\n", encoding="utf-8")
            metadata = import_cad(source, root / "assets", unit="mm")

        self.assertEqual(metadata["duplicate_triangles_removed"], 1)
        self.assertEqual(metadata["triangle_count"], len(cube.faces))
        self.assertTrue(metadata["watertight"])

    def test_paired_cad_fluid_and_solid_meshes_share_one_cht_grid(self):
        """Separate closed CAD volumes must classify without overlap."""
        fluid_volume = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
        fluid_volume.apply_translation([5.0, 5.0, 5.0])
        solid_volume = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
        solid_volume.apply_translation([15.0, 5.0, 5.0])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fluid_path = root / "fluid.stl"
            solid_path = root / "solid.stl"
            fluid_volume.export(fluid_path)
            solid_volume.export(solid_path)
            fluid_asset = import_cad(fluid_path, root / "assets", unit="mm")
            solid_asset = import_cad(solid_path, root / "assets", unit="mm")

            project = default_project()
            copper = dict(project["materials"][0])
            copper.update({"id": "copper", "name": "Copper"})
            project["materials"].append(copper)
            project["geometry"] = {
                "kind": "cad",
                "asset_id": fluid_asset["id"],
                "solid_asset_id": solid_asset["id"],
                "solid_material_id": "copper",
                "role": "fluid",
                "solids": [],
            }
            project["mesh"]["cells"] = [20, 10, 10]
            mesh = build_mesh(project, root / "assets")

        self.assertEqual(mesh["shape"], [20, 10, 10])
        self.assertEqual(int(mesh["fluid_mask"].sum()), 1000)
        self.assertEqual(int(mesh["solid_mask"].sum()), 1000)
        self.assertFalse(np.any(mesh["fluid_mask"] & mesh["solid_mask"]))
        self.assertTrue(np.all(mesh["thermal_mask"]))
        self.assertTrue(np.all(mesh["material_index"][mesh["solid_mask"]] == 1))
        self.assertEqual(mesh["geometry_metadata"]["fluid_asset_id"], fluid_asset["id"])
        self.assertEqual(mesh["geometry_metadata"]["solid_asset_id"], solid_asset["id"])

    def test_cad_patch_links_and_unresolved_selectors_are_bounded(self):
        cube = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "cube.stl"
            cube.export(source)
            metadata = import_cad(source, root / "assets", unit="mm")
            patch_id = metadata["surface_groups"][0]["id"]
            project = default_project()
            project["geometry"] = {"kind": "cad", "asset_id": metadata["id"], "role": "fluid"}
            project["mesh"]["cells"] = [10, 10, 10]
            project["boundaries"] = [
                {
                    "id": "patch-inlet",
                    "name": "Selected CAD patch",
                    "face": f"cad:{patch_id}",
                    "flow": {"type": "velocity", "velocity": [0.001, 0.0, 0.0]},
                    "thermal": {"type": "heat_flux", "value": 10.0},
                },
                {
                    "id": "missing-patch",
                    "name": "Missing CAD patch",
                    "face": "cad:does-not-exist",
                    "flow": {"type": "wall"},
                    "thermal": {"type": "adiabatic"},
                },
            ]
            mesh = build_mesh(project, root / "assets")
        self.assertEqual(mesh["boundary_links"][f"cad:{patch_id}"].shape, (6, 10, 10, 10))
        self.assertGreater(int(mesh["boundary_links"][f"cad:{patch_id}"].sum()), 0)
        self.assertEqual(int(mesh["boundary_links"]["cad:does-not-exist"].sum()), 0)
        self.assertGreater(int(mesh["boundary_links"]["cad"].sum()), 0)
        self.assertEqual(len(mesh["surface_groups"]), 6)
        fluid = mesh["fluid_mask"]
        cad_links = mesh["boundary_links"]["cad"]
        for direction in range(6):
            axis = direction // 2
            neighbour = np.zeros_like(fluid)
            if direction % 2:
                source = tuple(slice(1, None) if index == axis else slice(None) for index in range(3))
                target = tuple(slice(0, -1) if index == axis else slice(None) for index in range(3))
            else:
                source = tuple(slice(0, -1) if index == axis else slice(None) for index in range(3))
                target = tuple(slice(1, None) if index == axis else slice(None) for index in range(3))
            neighbour[target] = fluid[source]
            self.assertFalse(np.any(cad_links[direction] & neighbour))

    def test_tilted_cad_patch_maps_to_directional_links(self):
        box = trimesh.creation.box(extents=[12.0, 8.0, 10.0])
        rotation = trimesh.transformations.rotation_matrix(np.deg2rad(23.0), [0.0, 1.0, 0.0])
        box.apply_transform(rotation)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "tilted.obj"
            box.export(source)
            metadata = import_cad(source, root / "assets", unit="mm")
            tilted = next(group for group in metadata["surface_groups"] if not group["axis_aligned"])
            project = default_project()
            project["geometry"] = {"kind": "cad", "asset_id": metadata["id"], "role": "fluid"}
            project["mesh"]["cells"] = [14, 14, 14]
            project["boundaries"] = [
                {
                    "id": "tilted-wall",
                    "name": "Tilted wall",
                    "face": f"cad:{tilted['id']}",
                    "flow": {"type": "wall"},
                    "thermal": {"type": "adiabatic"},
                }
            ]
            mesh = build_mesh(project, root / "assets")
        links = mesh["boundary_links"][f"cad:{tilted['id']}"]
        self.assertGreater(int(links.sum()), 0)
        self.assertEqual(links.ndim, 4)
        self.assertEqual(links.shape[0], 6)

    def test_cad_obstacle_uses_computational_box_and_masks_volume(self):
        cube = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "obstacle.obj"
            # OBJ keeps shared indices, which exercises the non-STL path.
            cube.export(source)
            metadata = import_cad(source, root / "assets", unit="mm")
            project = default_project()
            project["geometry"] = {
                "kind": "cad",
                "asset_id": metadata["id"],
                "role": "obstacle",
                "size": [0.01, 0.01, 0.01],
            }
            project["mesh"]["cells"] = [10, 10, 10]
            mesh = build_mesh(project, root / "assets")
            self.assertEqual(mesh["shape"], [10, 10, 10])
            self.assertLess(int(mesh["fluid_mask"].sum()), 1000)
            self.assertGreater(int(mesh["fluid_mask"].sum()), 0)

    def test_cad_grid_keeps_isotropic_spacing_for_noninteger_bounds(self):
        box = trimesh.creation.box(extents=[13.0, 8.0, 8.0])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "rounded.obj"
            box.export(source)
            metadata = import_cad(source, root / "assets", unit="mm")
            project = default_project()
            project["geometry"] = {"kind": "cad", "asset_id": metadata["id"], "role": "fluid"}
            project["mesh"]["cells"] = [10, 10, 10]
            mesh = build_mesh(project, root / "assets")
        self.assertEqual(mesh["shape"], [10, 7, 7])
        self.assertTrue(any("isotropic spacing" in warning for warning in mesh["warnings"]))
        self.assertAlmostEqual(mesh["origin"][0], -0.0065)
        self.assertAlmostEqual(mesh["origin"][1], -0.00455)
        np.testing.assert_allclose(mesh["spacing"], [0.0013] * 3)
        self.assertTrue(mesh["fluid_mask"].all())

    def test_open_surface_is_rejected_for_fluid_volume(self):
        # Removing two faces leaves a real open surface; import stores it, but
        # volume classification must refuse to guess the missing closure.
        cube = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
        open_cube = trimesh.Trimesh(vertices=cube.vertices, faces=cube.faces[:-2], process=False)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "open.obj"
            open_cube.export(source)
            metadata = import_cad(source, root / "assets", unit="mm")
            project = default_project()
            project["geometry"] = {"kind": "cad", "asset_id": metadata["id"], "role": "fluid"}
            project["mesh"]["cells"] = [10, 10, 10]
            with self.assertRaisesRegex(ValueError, "watertight"):
                build_mesh(project, root / "assets")

    @unittest.skipIf(gmsh is None, "gmsh and its native CAD dependencies are optional")
    def test_step_is_tessellated_by_gmsh(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "box.step"
            gmsh.initialize(interruptible=False)
            try:
                gmsh.model.add("workbench-test-box")
                gmsh.model.occ.addBox(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
                gmsh.model.occ.synchronize()
                gmsh.write(str(source))
            finally:
                gmsh.finalize()
            metadata = import_cad(source, root / "assets", unit="mm")
            self.assertEqual(metadata["source_extension"], ".step")
            self.assertEqual(metadata["triangle_count"] > 0, True)
            self.assertGreaterEqual(len(metadata["surface_groups"]), 6)
            self.assertEqual(len(metadata["triangle_groups"]), metadata["triangle_count"])
            np.testing.assert_allclose(metadata["bounds"], [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
