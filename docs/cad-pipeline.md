# CAD import architecture

## Decision

Use a two-path CAD pipeline.

1. The browser previews STEP, IGES and BREP with `occt-import-js`, executed
   in a Web Worker.  The returned B-rep face ranges are the source of face
   picking and boundary-condition selection.
2. The server preserves the source CAD and separately validates, tessellates
   and voxelises a closed fluid volume for XLB.  A failed simulation mesh must
   never replace the preview geometry with a partial automatic repair.

This separates an interactive CAD view from the conservative geometry required
for a numerical domain.  They have different correctness requirements.

## Evaluated projects

| Project | Result | Reason |
| --- | --- | --- |
| `tegos/cad-3d-viewer` | Adopt its browser-worker architecture | It uses `occt-import-js`, preserves B-rep face ranges, and supports per-face picking. Its application code is MIT. |
| `kovacsv/occt-import-js` | Runtime dependency for the preview path | It imports STEP, IGES and BREP to structured mesh JSON in the browser. The dynamically loaded WASM remains LGPL-2.1 and ships with its license texts. |
| `fougue/mayo` | Optional future diagnostic/conversion tool | Mayo is a capable BSD-2-Clause Qt/OpenCascade desktop application, but embedding its GUI and C++ stack in the simulation container would add substantial maintenance without solving volume validity. |
| `donalffons/opencascade.js` | Do not adopt directly | It exposes a broad OpenCascade WASM API under LGPL-2.1. The narrower importer is sufficient and has a smaller integration surface. |
| `CadQuery` / OCP | Do not add to the primary image | It can import STEP and tessellate server-side, but its native OpenCascade/VTK dependency stack conflicts with the current slim CUDA image. Evaluate it later as a separate worker image if B-rep healing becomes necessary. |
| `CNCKitchen/meshStep` | Do not embed | It is AGPL-3.0-only, which is unsuitable for the current distribution model without a commercial agreement. |

## Acceptance rules

- Preview faces retain their original B-rep face IDs; smooth-facet grouping is
  a UI convenience only.
- The backend reports open edges, non-manifold edges and failed volume
  classification explicitly.
- Any automatic repair must preserve the CAD extents and must not silently
  discard a disconnected channel or a second flow pass.
- Simulation starts only after the server has a closed, consistently oriented
  fluid volume.

## Implemented preview path

The static Web Worker, upstream WASM/license assets, source-asset endpoint,
and B-rep face adapter are included in the workbench.  The current server
tessellation remains the calculation path until a separate topology-healing
worker is justified.
