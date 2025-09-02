import { watch } from "fs";

console.log("Building TypeScript files...");

// Build all TypeScript files in src directory
const result = await Bun.build({
  entrypoints: ['./src/app.ts'],
  outdir: './src',
  target: 'browser',
  format: 'esm',
  splitting: true,
  sourcemap: 'external',
  minify: false,
});

if (!result.success) {
  console.error("Build failed:", result.logs);
  process.exit(1);
}

console.log("✅ Build complete!");

// Watch mode if --watch flag is passed
if (process.argv.includes('--watch')) {
  console.log("👀 Watching for changes...");
  
  watch('./src', { recursive: true }, async (event, filename) => {
    if (filename?.endsWith('.ts') && !filename.endsWith('.js')) {
      console.log(`Rebuilding ${filename}...`);
      
      const result = await Bun.build({
        entrypoints: ['./src/app.ts'],
        outdir: './src',
        target: 'browser',
        format: 'esm',
        splitting: true,
        sourcemap: 'external',
        minify: false,
      });
      
      if (result.success) {
        console.log(`✅ Rebuilt successfully`);
      } else {
        console.error(`❌ Build failed:`, result.logs);
      }
    }
  });
}