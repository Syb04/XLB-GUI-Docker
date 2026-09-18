const fs = require('fs');
const path = require('path');
const vendor = path.join(__dirname, 'vendor', 'occt-import-js');
const init = require(path.join(vendor, 'occt-import-js.js'));
const source = process.argv[2];
const extension = String(process.argv[3] || path.extname(source)).toLowerCase();
const operation = extension === '.iges' || extension === '.igs' ? 'ReadIgesFile' : extension === '.brep' ? 'ReadBrepFile' : 'ReadStepFile';
init({ locateFile: (name) => path.join(vendor, name) }).then((occt) => {
  const result = occt[operation](fs.readFileSync(source), { linearUnit: 'meter', linearDeflectionType: 'bounding_box_ratio', linearDeflection: 0.001, angularDeflection: 0.2 });
  if (!result?.success || !Array.isArray(result.meshes)) throw new Error('OpenCascade could not read the CAD source');
  const vertices = [], faces = [], groups = [], names = {};
  result.meshes.forEach((mesh, meshIndex) => {
    const position = mesh?.attributes?.position?.array || [], index = mesh?.index?.array || [], offset = vertices.length;
    for (let point = 0; point + 2 < position.length; point += 3) vertices.push([position[point], position[point + 1], position[point + 2]]);
    const ranges = (mesh.brep_faces || []).map((face, faceIndex) => { const id = `brep-${meshIndex + 1}-${faceIndex + 1}`; names[id] = `CAD Face ${meshIndex + 1}.${faceIndex + 1}`; return { id, first: Number(face.first), last: Number(face.last) }; });
    for (let triangle = 0; triangle + 2 < index.length; triangle += 3) { const row = triangle / 3; const group = ranges.find((face) => row >= face.first && row <= face.last)?.id || `brep-${meshIndex + 1}-unclassified`; faces.push([offset + Number(index[triangle]), offset + Number(index[triangle + 1]), offset + Number(index[triangle + 2])]); groups.push(group); }
  });
  const remap = new Map(), merged = [], indices = vertices.map((point) => {
    const key = point.map((value) => Math.round(Number(value) * 1e9)).join(',');
    if (!remap.has(key)) { remap.set(key, merged.length); merged.push(point); }
    return remap.get(key);
  });
  process.stdout.write(JSON.stringify({ vertices: merged, faces: faces.map((face) => face.map((index) => indices[index])), groups, names }));
}).catch((error) => { process.stderr.write(String(error?.stack || error)); process.exitCode = 1; });
