const { onRequest } = require("firebase-functions/v2/https");
const { getConfig, handleClassify } = require("./ai");

function sendJson(res, status, payload) {
  res.status(status).set("Cache-Control", "no-store").json(payload);
}

function routePath(req) {
  const raw = req.originalUrl || req.url || req.path || "";
  return raw.split("?")[0];
}

exports.api = onRequest({ cors: true, maxInstances: 10 }, async (req, res) => {
  if (req.method === "OPTIONS") {
    res.set("Access-Control-Allow-Origin", "*");
    res.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set("Access-Control-Allow-Headers", "Content-Type");
    return res.status(204).send("");
  }

  const path = routePath(req);

  if (req.method === "GET" && path.includes("/api/ai/config")) {
    return sendJson(res, 200, getConfig());
  }

  if (req.method === "POST" && path.includes("/api/ai/classify")) {
    const result = await handleClassify(req.body || {});
    return sendJson(res, result.status, result.payload);
  }

  return sendJson(res, 404, { error: "Not found", path });
});
