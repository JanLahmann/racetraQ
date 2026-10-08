import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

const demoJs = fileURLToPath(new URL('../racetraq/web/js', import.meta.url));

// base './': the build is a static folder that works from any URL path
// (GitHub Pages project path, qamposer.org/racetraq/, a Pi's web server).
export default defineConfig({
  base: './',
  plugins: [react()],
  resolve: {
    // the server demo's renderer, imported unchanged
    alias: { '@demo': demoJs },
  },
  server: { fs: { allow: ['..'] } },
  test: {
    environment: 'jsdom', // @qamposer/react injects its styles on import
    include: ['tests/**/*.test.ts'],
  },
});
