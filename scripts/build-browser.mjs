import { build } from 'esbuild';

await build({ entryPoints: ['src/browser.ts', 'src/browser-worker.ts'], outdir: 'dist', bundle: true,
  platform: 'browser', format: 'esm', target: 'es2022', sourcemap: true });
