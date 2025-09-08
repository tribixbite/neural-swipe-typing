// Bundle the app with all dependencies
await Bun.build({
  entrypoints: ['./src/app.ts'],
  outdir: './public/src',
  target: 'browser',
  format: 'esm',
  splitting: false,
  sourcemap: 'external',
  minify: false,
  naming: '[name]-bundle.[ext]',
  external: ['onnxruntime-web'], // Mark as external since we're loading it via CDN
});

console.log('✅ Bundle created');