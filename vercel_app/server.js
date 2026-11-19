// Local server: no Vercel account needed. Run with:  node server.js
const http = require("http"), fs = require("fs"), path = require("path");

try { // read ANTHROPIC_API_KEY (and optional settings) from .env.local
  fs.readFileSync(path.join(__dirname, ".env.local"), "utf8").split("\n").forEach((l) => {
    const m = l.match(/^\s*([A-Z_]+)\s*=\s*(.*?)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
  });
} catch {}

const handler = require("./api/analyze");
const PUB = path.join(__dirname, "public");

http.createServer((req, res) => {
  if (req.url.startsWith("/api/analyze")) {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      try { req.body = JSON.parse(body || "{}"); } catch { req.body = {}; }
      res.status = (c) => { res.statusCode = c; return res; };
      res.json = (o) => { res.setHeader("content-type", "application/json"); res.end(JSON.stringify(o)); };
      handler(req, res);
    });
    return;
  }
  const f = path.join(PUB, req.url === "/" ? "index.html" : req.url.split("?")[0]);
  if (!f.startsWith(PUB) || !fs.existsSync(f) || !fs.statSync(f).isFile()) { res.statusCode = 404; return res.end("Not found"); }
  res.setHeader("content-type", f.endsWith(".html") ? "text/html; charset=utf-8" : "text/plain");
  res.end(fs.readFileSync(f));
}).listen(3000, () => console.log("Website running. Open http://localhost:3000  (Ctrl+C to stop)"));
