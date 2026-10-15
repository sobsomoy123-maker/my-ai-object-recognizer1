import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.esm.js';
import * as mobilenet from 'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/+esm';

let featureModelPromise;
export async function loadFeatureModel(onProgress = () => {}) {
  if (!featureModelPromise) featureModelPromise = (async () => {
    await tf.ready(); onProgress('Loading MobileNet feature extractor…');
    return mobilenet.load({version: 2, alpha: 1.0});
  })().catch(err => { featureModelPromise = null; throw err; });
  return featureModelPromise;
}
export function imageToTensor(image) { return tf.tidy(() => tf.browser.fromPixels(image).resizeBilinear([224,224]).toFloat().div(255).expandDims(0)); }
export async function extractFeature(model, image) {
  // MobileNet's infer() accepts an image/video/canvas element and performs its own preprocessing.
  const result = model.infer(image, true);
  try { const data = await result.data(); return Array.from(data); }
  finally { result.dispose(); }
}
export async function trainClassifier(featureModel, samples, onProgress = () => {}) {
  const labels = [...new Set(samples.map(s => s.category))].sort((a,b)=>a.localeCompare(b));
  if (labels.length < 2) throw new Error('Add training images to at least two categories before training.');
  const counts = Object.fromEntries(labels.map(l => [l, samples.filter(s => s.category === l).length]));
  if (labels.some(l => counts[l] < 2)) throw new Error('Each category needs at least 2 saved images. Add more examples, then retry.');
  const xs = [], ys = [];
  try {
    for (let i=0;i<samples.length;i++) {
      onProgress(8 + Math.round((i/samples.length)*35), `Extracting features ${i+1} of ${samples.length}…`);
      const img = await decodeImage(samples[i].dataUrl);
      try { xs.push(await extractFeature(featureModel,img)); } finally { img.close?.(); }
      ys.push(labels.indexOf(samples[i].category));
    }
    // When every category has at least 3 examples, reserve one example per category
    // as a small held-out validation set. With smaller datasets, use all examples.
    const heldOut = labels.every(label => counts[label] >= 3);
    const trainIndices = [], testIndices = [];
    for (const label of labels) {
      const ids = ys.map((y,i)=>y===labels.indexOf(label)?i:-1).filter(i=>i>=0);
      if (heldOut) testIndices.push(ids.pop());
      trainIndices.push(...ids);
    }
    const xTensor = tf.tensor2d(trainIndices.map(i=>xs[i]));
    const yTensor = tf.oneHot(tf.tensor1d(trainIndices.map(i=>ys[i]),'int32'),labels.length);
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape:[xs[0].length],units:128,activation:'relu'}));
    model.add(tf.layers.dropout({rate:0.2}));
    model.add(tf.layers.dense({units:labels.length,activation:'softmax'}));
    model.compile({optimizer:tf.train.adam(0.001),loss:'categoricalCrossentropy',metrics:['accuracy']});
    onProgress(48,'Training classifier…');
    const history = await model.fit(xTensor,yTensor,{epochs:35,batchSize:Math.min(16,trainIndices.length),shuffle:true,validationSplit:trainIndices.length>=10?0.15:0,callbacks:{onEpochEnd:(epoch,logs)=>onProgress(50+Math.round((epoch+1)/35*38),`Training epoch ${epoch+1}/35 · loss ${Number(logs.loss).toFixed(3)}`)}});
    let validationAccuracy = null;
    if (heldOut) {
      onProgress(90,'Evaluating on held-out examples…');
      const testX = tf.tensor2d(testIndices.map(i=>xs[i]));
      const predictions = model.predict(testX);
      const values = await predictions.data();
      let correct = 0;
      for (let row=0;row<testIndices.length;row++) {
        let best=0; for(let col=1;col<labels.length;col++) if(values[row*labels.length+col]>values[row*labels.length+best])best=col;
        if(best===ys[testIndices[row]])correct++;
      }
      validationAccuracy = correct/testIndices.length;
      predictions.dispose(); testX.dispose();
    }
    const finalAccuracy = validationAccuracy ?? history.history.acc?.at(-1) ?? history.history.accuracy?.at(-1) ?? null;
    xTensor.dispose(); yTensor.dispose();
    return {model,labels,accuracy:typeof finalAccuracy==='number'?finalAccuracy:null,validationAccuracy,heldOutCount:testIndices.length,sampleCount:samples.length,categoryCounts:counts};
    return {model,labels,accuracy:typeof finalAccuracy==='number'?finalAccuracy:null,sampleCount:samples.length,categoryCounts:counts};
  } catch(e) { throw e; }
}
export async function predictFeature(model, labels, feature) {
  const input = tf.tensor2d([feature]);
  try { const out = model.predict(input); const values = await out.data(); out.dispose(); const ranked=Array.from(values).map((confidence,i)=>({category:labels[i],confidence})).sort((a,b)=>b.confidence-a.confidence); return ranked[0] || null; }
  finally { input.dispose(); }
}
export async function serializeModel(model) {
  let topology, weightSpecs, weightData;
  await model.save(tf.io.withSaveHandler(async artifacts => { topology=artifacts.modelTopology; weightSpecs=artifacts.weightSpecs; weightData=artifacts.weightData; return {modelArtifactsInfo:{dateSaved:new Date(),modelTopologyType:'JSON',modelTopologyBytes:JSON.stringify(topology).length,weightSpecsBytes:JSON.stringify(weightSpecs).length,weightDataBytes:weightData.byteLength}}; }));
  const bytes = new Uint8Array(weightData); let binary=''; for(let i=0;i<bytes.length;i+=0x8000) binary+=String.fromCharCode(...bytes.subarray(i,i+0x8000));
  return {format:'tfjs-layers-model',modelTopology:topology,weightSpecs,weightDataBase64:btoa(binary)};
}
export async function loadSavedClassifier(artifact) {
  const binary=atob(artifact.weightDataBase64); const bytes=new Uint8Array(binary.length); for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
  return tf.loadLayersModel(tf.io.fromMemory({modelTopology:artifact.modelTopology,weightSpecs:artifact.weightSpecs,weightData:bytes.buffer}));
}
export function decodeImage(dataUrl) { return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('A saved image could not be decoded.'));img.src=dataUrl;}); }
export async function fileToDataUrl(file) { if(!['image/jpeg','image/png','image/webp'].includes(file.type)) throw new Error(`${file.name}: use JPEG, PNG, or WebP.`); if(file.size>2*1024*1024) throw new Error(`${file.name}: image is larger than 2 MB.`); const url=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=()=>reject(new Error('Could not read selected file.'));r.readAsDataURL(file)}); const img=await decodeImage(url); try {if(img.width<64||img.height<64||img.width>4096||img.height>4096)throw new Error(`${file.name}: image dimensions must be 64–4096 px on each side.`);} finally {img.close?.();} return url; }
