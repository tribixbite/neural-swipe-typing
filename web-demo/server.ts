import { join } from "path";

const PORT = process.env.PORT || 3456;

console.log(`🚀 Starting Swipe Typing Demo server on port ${PORT}...`);

Bun.serve({
  port: PORT,
  async fetch(req) {
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
    
    // Serve and transpile TypeScript/JavaScript files
    if (url.pathname.startsWith("/src/")) {
      const filePath = join(import.meta.dir, url.pathname.slice(1));
      
      // If requesting .js but .ts exists, transpile the TypeScript
      if (url.pathname.endsWith('.js')) {
        const tsPath = filePath.replace(/\.js$/, '.ts');
        const tsFile = Bun.file(tsPath);
        
        if (await tsFile.exists()) {
          // Transpile TypeScript to JavaScript
          const transpiler = new Bun.Transpiler({
            loader: 'ts',
            target: 'browser',
            tsconfig: {
              compilerOptions: {
                target: "ES2020",
                module: "ES2020",
                lib: ["ES2020", "DOM"],
                moduleResolution: "node",
              }
            }
          });
          
          const code = await tsFile.text();
          const result = await transpiler.transform(code);
          
          return new Response(result, {
            headers: {
              "Content-Type": "application/javascript",
              "Access-Control-Allow-Origin": "*"
            }
          });
        }
      }
      
      // Serve file as-is if it exists
      const file = Bun.file(filePath);
      if (await file.exists()) {
        const contentType = filePath.endsWith('.ts') ? 'application/typescript' : 'application/javascript';
        return new Response(file, {
          headers: {
            "Content-Type": contentType,
            "Access-Control-Allow-Origin": "*"
          }
        });
      }
    }
    
    // Serve files from public directory
    if (url.pathname.startsWith("/src/") || url.pathname === "/" || url.pathname === "") {
      let filePath: string;
      
      if (url.pathname === "/" || url.pathname === "") {
        filePath = join(import.meta.dir, "public", "index.html");
      } else {
        filePath = join(import.meta.dir, "public", url.pathname.slice(1));
      }
      
      const file = Bun.file(filePath);
      if (await file.exists()) {
        let contentType = "text/plain";
        if (filePath.endsWith(".html")) contentType = "text/html; charset=utf-8";
        else if (filePath.endsWith(".js")) contentType = "application/javascript";
        else if (filePath.endsWith(".css")) contentType = "text/css";
        else if (filePath.endsWith(".json")) contentType = "application/json";
        
        return new Response(file, {
          headers: {
            "Content-Type": contentType,
            "Access-Control-Allow-Origin": "*"
          }
        });
      }
    }
    
    return new Response("Not found", { status: 404 });
  },
  development: {
    hmr: true,
    console: true,
  }
});

console.log(`✅ Server running at http://localhost:${PORT}`);