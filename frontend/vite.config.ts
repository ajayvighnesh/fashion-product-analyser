/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // set BASE_PATH=/fashion-product-analyser/ when hosting on GitHub Pages
  base: process.env.BASE_PATH ?? "/",
  server: {
    // during development, /api calls go to the local FastAPI server
    proxy: { "/api": "http://localhost:8000" },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/setupTests.ts"],
  },
});
