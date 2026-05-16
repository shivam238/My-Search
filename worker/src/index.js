import { initEnv, getConfig, handleClassify } from "./ai.js";

const cors = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Cache-Control": "no-store",
};

function json(status, payload) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { ...cors, "Content-Type": "application/json" },
  });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }

    initEnv(env);
    const path = new URL(request.url).pathname;

    if (request.method === "GET" && path.endsWith("/api/ai/config")) {
      return json(200, getConfig());
    }

    if (request.method === "POST" && path.endsWith("/api/ai/classify")) {
      let body = {};
      try {
        body = await request.json();
      } catch (_) {}
      const result = await handleClassify(body);
      return json(result.status, result.payload);
    }

    return json(404, { error: "Not found", path });
  },
};
