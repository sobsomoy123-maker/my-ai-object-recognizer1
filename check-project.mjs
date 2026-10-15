import {readFileSync,existsSync} from 'node:fs';
const required=['index.html','package.json','netlify.toml','src/app.js','src/train.js','src/style.css','netlify/functions/save-sample.mjs','netlify/functions/list-samples.mjs','netlify/functions/save-model.mjs','netlify/functions/get-model.mjs','netlify/functions/_shared.mjs'];
const missing=required.filter(f=>!existsSync(f));if(missing.length){console.error('Missing required files:',missing.join(', '));process.exit(1)}
const html=readFileSync('index.html','utf8');const app=readFileSync('src/app.js','utf8');const pkg=JSON.parse(readFileSync('package.json','utf8'));
for(const id of ['camera-video','sample-form','train-model','library-grid'])if(!html.includes(`id="${id}"`))throw new Error(`Missing UI element ${id}`);
for(const route of ['/save-sample','/list-samples','/save-model','/get-model'])if(!app.includes(route))throw new Error(`Frontend route mismatch: ${route}`);
if(!pkg.scripts.dev||!pkg.scripts.build||!pkg.dependencies['@netlify/blobs'])throw new Error('Missing required package configuration');
console.log(`Project structure check passed (${required.length} required files).`);