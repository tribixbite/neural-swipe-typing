import index from "./index.html";
import { join } from "path";

const PORT = process.env.PORT || 3456;

console.log(`🚀 Starting Swipe Typing Demo server on port ${PORT}...`);

Bun.serve({
  port: PORT,
  fetch(req) {
    const url = new URL(req.url);
    
    // Serve static files from deployment_package
    if (url.pathname.startsWith("/models/")) {
      const filename = url.pathname.slice(8); // Remove "/models/"
      const modelPath = join(import.meta.dir, "../deployment_package", filename);
      const file = Bun.file(modelPath);
      
      if (file.size > 0) {
        return new Response(file, {
          headers: {
            "Content-Type": filename.endsWith(".onnx") ? "application/octet-stream" : "application/json",
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
          }
        });
      }
      return new Response("Model not found", { status: 404 });
    }
    
    // Serve main app
    if (url.pathname === "/" || url.pathname === "") {
      return new Response(index, {
        headers: {
          "Content-Type": "text/html; charset=utf-8"
        }
      });
    }
    
    return new Response("Not found", { status: 404 });
  },
  development: {
    hmr: true,
    console: true,
  }
});

console.log(`✅ Server running at http://localhost:${PORT}`);