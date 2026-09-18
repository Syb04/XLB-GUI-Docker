/* CAD preview worker.  OpenCascade runs off the UI thread so an assembly
 * import cannot freeze camera interaction or the model-builder controls. */
importScripts('/vendor/occt-import-js/occt-import-js.js');

let occtReady = null;

function importer() {
  if (!occtReady) occtReady = occtimportjs({ locateFile: (name) => `/vendor/occt-import-js/${name}` });
  return occtReady;
}

self.onmessage = async (event) => {
  const { id, extension, buffer } = event.data || {};
  try {
    const occt = await importer();
    const source = new Uint8Array(buffer);
    const params = {
      linearUnit: 'meter',
      linearDeflectionType: 'bounding_box_ratio',
      linearDeflection: 0.001,
      angularDeflection: 0.2
    };
    const operation = extension === '.iges' || extension === '.igs'
      ? 'ReadIgesFile'
      : extension === '.brep' ? 'ReadBrepFile' : 'ReadStepFile';
    const result = occt[operation](source, params);
    if (!result?.success || !Array.isArray(result.meshes)) throw new Error('OpenCascade could not read this CAD file');
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};
