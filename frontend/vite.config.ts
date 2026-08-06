import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273,
    // The API key never reaches the browser bundle. Requests go to the dev
    // server, which attaches the credential server-side -- a key shipped in
    // client JavaScript is readable by anyone who opens devtools.
    proxy: {
      "/api": {
        target: process.env.AGENTLENS_API_URL ?? "http://127.0.0.1:8100",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => {
            const key = process.env.AGENTLENS_API_KEY;
            if (key) proxyReq.setHeader("x-agentlens-api-key", key);
          });
        },
      },
    },
  },
});
