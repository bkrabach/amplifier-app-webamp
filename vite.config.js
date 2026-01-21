import { defineConfig } from 'vite';

export default defineConfig({
  // Required for SharedArrayBuffer (needed by some WebGPU features)
  server: {
    host: '0.0.0.0',  // Listen on all interfaces for remote access
    allowedHosts: ['spark-1', 'spark-1.local', 'localhost'],
    // COOP/COEP headers - needed for SharedArrayBuffer but may interfere with WebGPU detection
    // Temporarily disabled for debugging
    // headers: {
    //   'Cross-Origin-Embedder-Policy': 'require-corp',
    //   'Cross-Origin-Opener-Policy': 'same-origin',
    // },
  },
  build: {
    target: 'esnext',
  },
  optimizeDeps: {
    exclude: ['pyodide'],
  },
});
