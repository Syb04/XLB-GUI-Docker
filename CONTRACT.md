# Workbench implementation contract

## RAM-backed CUDA collision

`study.device="cuda:0-ram"` selects the explicit RAM pipeline regardless of `XLB_COMPUTE_BACKEND`. Full lattice state, streaming, boundaries, thermal, LES and properties stay on CPU. `ram_backend.RamCollision` batches local XLB equilibrium/BGK collision on CUDA; output returns to CPU before global streaming, so no artificial inter-batch boundaries are introduced. Optional `study.gpu_batch_cells` defaults to 65536, validated as an integer in [1024,1048576], preserved for all devices, excluded from restart physics fingerprint. CUDA availability is required; no silent CPU fallback. Diagnostics record the actual state/collision devices and offload statistics. Existing `cpu` and `cuda:0` semantics are unchanged. See `docs/ram-compute.md`.

## Material catalog and CSV import

`GET /api/materials/presets` serves the bundled catalog at `workbench/data/material_presets.json`. Presets include a schema-compatible material body (without ID), label, description, pressure/range, source links and generation provenance. They are explicitly applied or added in the material editor; existing inputs remain unchanged until the user applies them. Catalog values are generated with pinned CoolProp in a development script; no runtime dependency is added.

`POST /api/materials/csv` accepts `{text,property_key?}` and returns `{properties,rows,temperature_unit,warnings}` without saving anything. `properties` contains canonical SI table descriptors; the UI previews and merges these into the captured material only when explicitly applied, with stale-selection guards. A wide header has one `temperature_K`/`temperature_C` column plus named property columns. A two-column `temperature_*,value` form needs a canonical `property_key`. Unknown columns, duplicate temperatures, non-finite/non-positive values, malformed rows and size limits are rejected. See `docs/material-properties.md` and `workbench/material_csv.py` for the format. Project schema, solver, and checkpoint formats are unchanged.

## Continuation / restart extension

`simulate(..., restart_from=None)` optionally loads a versioned `restart.npz` containing evolved FP32 D3Q27 distributions and FP64 temperature/internal state. Complete and gracefully stopped runs atomically publish a final checkpoint. Plot snapshots are not restart checkpoints. Compatibility validates the physical input, dt, effective mesh, field shapes and precision before stepping. Opaque CAD asset IDs are excluded from the input fingerprint so archive import can remap them; the effective mesh fingerprint still protects geometry and CAD boundary links.

`GET /api/runs/:id/restart` returns `{available,reason?}` or `{available:true,step,dt,time}` by reading small metadata only. `POST` on the same endpoint accepts only `{additional_steps: positive integer}` and returns a new queued run (202). It reuses immutable source input and mesh, keeps dt unchanged, and sets total `study.steps = checkpoint.step + additional_steps`. Source results and saved project edits remain unchanged. Status includes `restart_from`, `start_step`, `end_step`, `additional_steps`; progress refers to the additional range while history/frame steps and times remain absolute. The new run starts with a frame at the source step when snapshots are enabled. Missing legacy checkpoints and active/failed/interrupted runs cannot be continued. Archive export includes checkpoints, and import remaps lineage IDs within the archive. Continuations remain independently restartable once their own checkpoint is saved.

## Total-cell mesh extension

`mesh.target_cells` selects automatic isotropic resolution from one positive integer, bounded by the configured total-cell limit. It takes precedence over legacy `mesh.cells`. Omitting it preserves existing axis counts. The target counts all grid cells, including solids and inactive exterior cells; integer dimensions mean actual totals can differ. Physical box dimensions are preserved exactly, with an explicit error if no isotropic integer subdivision fits the limits. CAD fluid grids retain centered exterior padding and at least three cells per axis.

`POST /api/mesh/estimate` accepts a project and returns `{shape,total_cells,origin,spacing,target_cells,warnings}` without allocating voxel masks. It uses the same grid planning as mesh generation, including actual CAD bounds and obstacle computational-box descriptors. The UI presents one target input and read-only calculated dimensions. Saved projects and immutable run inputs preserve the target so regenerated meshes use the same rule.

## Transient results extension

`study.snapshot_interval` is an optional non-negative integer: omitted/0 preserves final-only output; new projects default to 20; 1 saves every step. With snapshots enabled, step 0, each requested interval and the actual final/stopped step are saved. This is independent of `output_interval` (progress/history sampling). Requested checkpoints perform an additional batched device download; all other steps retain scalar-only transfers. `full_field_transfers` counts download batches, including checkpoints and finalization. Diagnostics include `snapshot_count`, `snapshot_bytes`, and `timings.snapshots_s` (intermediate transfer/write cost; final snapshot cost is in `save_s`).

`frames/static.npz` contains immutable masks, material indices, origin and spacing. `frames/step_000000020.npz` stores SI velocity, gauge pressure, temperature and kinematic eddy viscosity in FP32. Files are atomically renamed before `frames.json` publishes `{frames:[{step,time,file}]}`. Uncompressed snapshots consume about 24 bytes/cell/frame. The final `fields.npz`/VTK interface remains supported. See `docs/transient-results.md` for controls, definitions and storage limits. The previous final-download-only description below applies when snapshots are disabled.

Independent Python package `workbench`, plain JS/CSS frontend in `static`. No imports from legacy GUI.

Project JSON (SI): `{schema_version:1,name,geometry:{kind:'box'|'cad',size:[0.06,0.02,0.02],asset_id:null,role:'fluid'|'obstacle'},materials:[{id:'water',name:'Water',density:{kind:'constant',value:998},viscosity:{kind:'constant',value:0.001},heat_capacity:{kind:'constant',value:4182},conductivity:{kind:'constant',value:0.6}}],physics:{flow:true,thermal:true,material_id:'water',initial_temperature:293.15},boundaries:[{id:'inlet',name:'Inlet',face:'xmin',flow:{type:'velocity',velocity:[0.01,0,0]},thermal:{type:'temperature',value:293.15}},{id:'outlet',name:'Outlet',face:'xmax',flow:{type:'pressure',value:0},thermal:{type:'adiabatic'}}],mesh:{cells:[30,10,10]},study:{steps:200,output_interval:20,dt:0.01,device:'cpu'}}`.

All six box faces xmin/xmax/ymin/ymax/zmin/zmax may have one BC; omitted faces default stationary no-slip/adiabatic. CAD wall surface selector `cad` permitted. Temperature property is `{kind:'table',points:[[273.15,value],[373.15,value]]}` with linear interpolation, clamped endpoints and explicit warnings; no arbitrary eval.

schema.py: `default_project() -> dict`, `validate_project(project) -> dict` normalized or ValueError; `evaluate_property(prop, temperature)` scalar/ndarray.
geometry.py: `import_cad(path, asset_dir, unit='m') -> dict` (metadata containing id, vertices, faces, bounds, original_name), `build_mesh(project, assets_dir) -> dict` with `fluid_mask` numpy bool (nx,ny,nz), `origin` float 3-vector, `spacing` float 3-vector, `shape` list; optional surface vertices/faces. Cartesian isotropic cells required by solver: box size / cell count equal across axes (validation). CAD fluid volume uses CAD bounds, optional manual computational box for obstacle. Store imports under asset_dir/id/metadata.json, source extension, surface.npz. `build_mesh` retrieves asset by geometry.asset_id. CAD fluid spacing is rounded isotropically with symmetric padding around its bounds; obstacle box defaults to geometry.size centered on the CAD bounds. See examples/heated_channel.json for the full starter model.

solver.py: `simulate(project, mesh, run_dir, on_progress=None, should_stop=None) -> dict`; saves fields.npz (`velocity` shape (3,nx,ny,nz), `pressure`, `temperature`, `fluid_mask`, `origin`, `spacing`), history.csv and fields.vtk; on_progress JSON dict with step,progress,residual,temperature_min,temperature_max; returns status completed/stopped, steps, converged and diagnostics. XLB real operators mandatory, CPU default and optional cuda:0. Thermal advection diffusion may be separate FV scheme, real local temperature-dependent viscosity/conductivity/cp. Density reference incompressible explicit semantics.

Device pipeline: `XLB_COMPUTE_BACKEND=auto` uses the JAX-resident pipeline for `cuda:0` and NumPy reference thermal for `cpu`; `device` or `reference` overrides implementation but never overrides the requested hardware. `fields_jax.py` owns material/LES factories, `thermal_jax.py` owns FV step/stability factories. All mutable fields stay on device between steps; progress uses scalar reductions and final artifacts use one batched download. LBM storage is FP32; thermal/material/LES arithmetic is FP64. Diagnostics add `compute_backend`, `thermal_backend`, `les_backend`, `property_backend`, `full_field_transfers`, `device_scalar_transfers`, and `timings` (`first_step_s` includes compilation and execution, `warm_compute_s` excludes first step and progress sampling, `save_s` includes final evaluation/download/output). Run provenance hashes all workbench Python modules and records the backend environment selection.

HTTP API parent implements: GET /api/health; GET /api/default; GET /api/projects -> {projects:[{id,name,updated_at}]}; POST /api/projects body project -> {id,project}; GET /api/projects/:id -> {id,project,runs:[...]}; PUT /api/projects/:id body project; POST /api/import?filename=...&unit=mm raw bytes -> asset metadata; POST /api/mesh body project -> {shape,fluid_cells,total_cells,origin,spacing,preview:{points:[[x,y,z],...],vertices,faces},warnings}; POST /api/runs body {project_id,project} -> {id,status}; GET /api/runs/:id -> metadata including history progress; POST /api/runs/:id/stop; GET /api/runs/:id/slice?field=temperature|speed|pressure&axis=z&index=5 -> {values:2D array,min,max,unit,axis,index,shape,extent}; GET /api/runs/:id/files/:filename; GET /api/projects/:id/export -> zip entire project/assets/runs; POST /api/projects/import raw zip -> {id,project}. GET /api/assets/:id -> metadata. Errors JSON {error:string}.

Every run snapshots input; mesh regenerated from snapshot; global single worker queue. IDs generated server-side. Status failed distinct from completed; no fake fields. Frontend save before run; server handles unsaved project by create. Canvas 3D orbit geometry/mesh, canvas 2D result slice and convergence plot; no CDN dependencies.

## Residual monitor extension

Optional `study.monitor={tolerance:1e-5,consecutive_samples:5}` selects the relative field-change threshold and consecutive eligible samples (integer 1..100). It is excluded from restart physical fingerprints. Omitted settings use defaults. `output_interval` is the comparison interval, counted from the restart step. No auto-stop is performed. New history/progress samples carry `residual_kind:'sample_change_linf_v1'`, `time`, `sample_steps`, `sample_time`, `velocity_residual`, `pressure_residual`, `temperature_residual`, SI `velocity_change_max`, `pressure_change_max`, `temperature_change_max`, plus `monitor_tolerance`, `monitor_required_samples`, `monitor_consecutive_samples`, `monitor_satisfied`, and `monitor_eligible`. Baseline and disabled-field values are null. `residual` is the maximum active relative change. Each field must pass for the required number of full intervals; a partial final interval resets the consecutive count. `converged` now means this sustained stationarity criterion and is false on user stop. Diagnostics include `convergence_monitor`. The six reductions stay on the selected device until scalar download; RAM mode reduces on CPU. CSV is appended at every sample, status retains at most 2000 rows. See `docs/residual-monitor.md` for normalization, legacy semantics and limits.

## Gravity extension

Optional `physics.gravity` is `{enabled:false,mode:'uniform'|'buoyancy',vector:[0,0,-9.80665],reference_temperature:293.15}`. Omission remains omitted during normalization for legacy compatibility. The vector is SI acceleration; the reference temperature is kelvin. Flow must be enabled. Uniform mode applies full gravity. Buoyancy applies `(density(T)/density(Tref)-1)*g`, uses the reference density for LBM scaling, and reports reduced pressure with reference hydrostatic pressure removed. Density contrasts above 10% are rejected. Pressure BC values have the same gauge/reduced semantics as results. Guo forcing and physical half-step velocity correction apply on CPU, CUDA and RAM-batched collision; solid cells are masked. Diagnostics include `gravity` and `pressure_kind`. Disabled gravity is omitted from the restart physical fingerprint, enabled settings are included. Mesh estimates add nullable `force_dt_max` and `force_dt_basis` and include gravity constraints in `recommended_dt`. See `docs/gravity.md` for limits and verification.

## Turbulence and conjugate heat transfer extension

`physics.turbulence` is optional. Its normalized form is
`{model:'laminar'|'smagorinsky',smagorinsky_constant:0.17,turbulent_prandtl:0.9}`;
the constants are finite and positive and are retained for both models. An
omitted descriptor is normalized to the laminar defaults.

`geometry.solids` is an optional array of non-overlapping axis-aligned regions
inside a known computational box. Each entry is
`{id,name,origin:[m],size:[m],material_id}` and its material reference must be
present in `materials`. A CAD obstacle may additionally set
`geometry.solid_material_id` for the imported interior. Mesh construction
returns `solid_mask`, `thermal_mask`, and `material_index` arrays with the
fluid-mask shape. Material indices follow the project material order;
`-1` means inactive or a geometric solid without an assigned CHT material.

CAD boundaries may select an individual imported surface with
`face:'cad:<patch_id>'`. Individual selectors accept wall, velocity, or
pressure flow conditions and all thermal conditions. The global `face:'cad'`
selector is the fallback for surface links left by individual selectors.
STL/OBJ imports derive connected coplanar patches (`patch-1`, ...); STEP,
IGES, and BREP imports retain native gmsh surface tags (`surface-<tag>`). The
asset manifest exposes `surface_groups` entries with `id`, `name`,
`triangle_count`, `area`, `normal`, `centroid`, `bounds`, and `native_tag` when
available, plus `triangle_groups`, one patch id per normalized triangle.

For CAD meshes, `boundary_links` maps each requested selector and `cad` to a
boolean array of shape `(6,nx,ny,nz)`, ordered x-, x+, y-, y+, z-, z+. Links
are generated only for exposed fluid cells and are assigned to the nearest
actual surface triangle, including a surface coincident with a computational
domain border. Unresolved `cad:<patch_id>` selectors are emitted as all-false
arrays. Box-face keys use the same layout and mark fluid cells on the six
domain borders. `boundary_normals` provides the corresponding box and patch
normals where available.
