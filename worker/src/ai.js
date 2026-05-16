const VALID_TAGS = new Set(["movie", "anime", "game", "show", "song"]);
const PROVIDER_ENV = {
  gemini: "GEMINI_API_KEY",
  openai: "OPENAI_API_KEY",
  groq: "GROQ_API_KEY",
  grok: "GROK_API_KEY",
  claude: "ANTHROPIC_API_KEY",
};
const GEMINI_MODELS = [
  "gemini-2.5-flash",
  "gemini-2.0-flash",
  "gemini-flash-latest",
  "gemini-1.5-flash",
];

let ENV = {};

export function initEnv(env) {
  ENV = env || {};
}

function defaultProvider() {
  const p = (ENV.AI_PROVIDER || "gemini").toLowerCase();
  return PROVIDER_ENV[p] ? p : "gemini";
}

function providerKeys(name) {
  const envName = PROVIDER_ENV[name];
  if (!envName) return [];
  const raw = ENV[envName] || "";
  return raw.split(/[,;]+/).map((k) => k.trim()).filter(Boolean);
}

function configuredProviders() {
  return Object.keys(PROVIDER_ENV).filter((p) => providerKeys(p).length > 0);
}

function totalEnvKeyCount() {
  return Object.keys(PROVIDER_ENV).reduce((n, p) => n + providerKeys(p).length, 0);
}

function classifyPrompt(q) {
  return `You are a STRICT content classifier. Reply ONLY with valid JSON, no markdown, no explanation, no extra text.
Format: {"tags":["game"]}
Available tags: movie, anime, game, show, song
Rules:
- VIDEO GAME (any title, sequel, franchise, mobile/PC/console game) = ["game"] ONLY. Never add "movie".
- ANIME / MANGA / Japanese animation = ["anime"] ONLY
- MOVIE / FILM (theatrical release, not a game) = ["movie"] ONLY
- TV show / web series / season / episode = ["show"] ONLY
- SONG / MUSIC / album / artist / lyrics / track / single = ["song"] ONLY
- Never return all 5 tags. Default to a single most-likely tag.
Classify this: "${q}"`;
}

function extractTags(text) {
  if (!text) return null;
  let t = String(text).replace(/```[a-z]*|```/gi, "").trim();
  try {
    const parsed = JSON.parse(t);
    if (Array.isArray(parsed?.tags)) return parsed.tags;
    if (Array.isArray(parsed)) return parsed;
  } catch (_) {}
  const mobj = t.match(/\{[\s\S]*?\}/);
  if (mobj) {
    try {
      const parsed = JSON.parse(mobj[0]);
      if (Array.isArray(parsed?.tags)) return parsed.tags;
    } catch (_) {}
  }
  const amobj = t.match(/\[[\s\S]*?\]/);
  if (amobj) {
    try {
      const parsed = JSON.parse(amobj[0]);
      if (Array.isArray(parsed)) return parsed;
    } catch (_) {}
  }
  const found = [...VALID_TAGS].filter((v) => new RegExp(`\\b${v}\\b`, "i").test(t));
  return found.length ? found : null;
}

function cleanTags(tags) {
  if (!Array.isArray(tags)) return null;
  const cleaned = [...new Set(tags.map((t) => String(t).toLowerCase().trim()).filter((t) => VALID_TAGS.has(t)))];
  if (!cleaned.length || cleaned.length === 5) return null;
  return cleaned;
}

function isKeyFailureError(msg) {
  const s = String(msg || "").toLowerCase();
  return /quota|rate.?limit|billing|invalid.*api|api[_ ]?key|exhausted|permission|denied|unauthorized|exceeded|resource_exhausted|credit|401|403|429|402/.test(s);
}

async function httpJson(method, url, headers, body) {
  const opts = { method, headers: { ...headers } };
  if (body != null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(url, opts);
  const text = await resp.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { error: { message: text || resp.statusText } };
  }
  if (!resp.ok) {
    const errObj = data?.error ?? data;
    const msg = typeof errObj === "object" ? errObj.message || JSON.stringify(errObj) : String(errObj);
    throw new Error(msg || `HTTP ${resp.status}`);
  }
  return data;
}

async function classifyOpenAICompat(q, apiKey, baseUrl, model, jsonMode = true) {
  const body = {
    model,
    messages: [{ role: "user", content: classifyPrompt(q) }],
    max_tokens: 60,
    temperature: 0,
  };
  if (jsonMode) body.response_format = { type: "json_object" };
  const data = await httpJson("POST", `${baseUrl}/chat/completions`, { Authorization: `Bearer ${apiKey}` }, body);
  if (data?.error) {
    const err = data.error;
    throw new Error(typeof err === "object" ? err.message || String(err) : String(err));
  }
  const text = data?.choices?.[0]?.message?.content || "";
  return cleanTags(extractTags(text));
}

async function classifyGemini(q, apiKey) {
  const body = {
    contents: [{ parts: [{ text: classifyPrompt(q) }] }],
    generationConfig: { temperature: 0, maxOutputTokens: 60, responseMimeType: "application/json" },
  };
  let lastErr = null;
  for (const model of GEMINI_MODELS) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
    try {
      const data = await httpJson("POST", url, {}, body);
      if (data?.error) {
        const msg = data.error.message || "Gemini error";
        if (/not found|not supported/i.test(msg)) {
          lastErr = new Error(msg);
          continue;
        }
        throw new Error(msg);
      }
      const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
      return cleanTags(extractTags(text));
    } catch (e) {
      lastErr = e;
      if (!/not found|not supported|404/i.test(e.message || "")) throw e;
    }
  }
  if (lastErr) throw lastErr;
  return null;
}

async function classifyClaude(q, apiKey) {
  const data = await httpJson(
    "POST",
    "https://api.anthropic.com/v1/messages",
    { "x-api-key": apiKey, "anthropic-version": "2023-06-01" },
    {
      model: "claude-3-haiku-20240307",
      max_tokens: 60,
      messages: [{ role: "user", content: classifyPrompt(q) }],
    }
  );
  const text = data?.content?.[0]?.text || "";
  return cleanTags(extractTags(text));
}

async function classifyQueryWithKey(q, provider, apiKey) {
  switch (provider) {
    case "gemini":
      return classifyGemini(q, apiKey);
    case "openai":
      return classifyOpenAICompat(q, apiKey, "https://api.openai.com/v1", "gpt-4o-mini");
    case "groq":
      return classifyOpenAICompat(q, apiKey, "https://api.groq.com/openai/v1", "llama-3.1-8b-instant");
    case "grok":
      return classifyOpenAICompat(q, apiKey, "https://api.x.ai/v1", "grok-2-1212", false);
    case "claude":
      return classifyClaude(q, apiKey);
    default:
      throw new Error(`Unknown provider: ${provider}`);
  }
}

async function classifyWithEnvFallback(q, preferredProvider) {
  const errors = [];
  const order = [preferredProvider, ...Object.keys(PROVIDER_ENV).filter((p) => p !== preferredProvider)];
  for (const prov of order) {
    const keys = providerKeys(prov);
    for (let idx = 0; idx < keys.length; idx++) {
      try {
        const tags = await classifyQueryWithKey(q, prov, keys[idx]);
        if (tags?.length) {
          return { tags, provider: prov, source: "env", keyIndex: idx };
        }
      } catch (e) {
        errors.push(`${prov}[${idx}]: ${e.message}`);
      }
    }
  }
  throw new Error(errors[errors.length - 1] || "No env API keys configured");
}

export function getConfig() {
  const provider = defaultProvider();
  return {
    configured: totalEnvKeyCount() > 0,
    provider,
    providers: configuredProviders(),
    keyCount: totalEnvKeyCount(),
  };
}

export async function handleClassify(body) {
  const q = (body.query || body.q || "").trim();
  if (!q) return { status: 400, payload: { error: "query is required" } };

  let provider = (body.provider || defaultProvider()).toLowerCase();
  if (!PROVIDER_ENV[provider]) return { status: 400, payload: { error: `invalid provider: ${provider}` } };

  if (totalEnvKeyCount() === 0) {
    return { status: 503, payload: { error: "No API keys in .env" } };
  }

  try {
    const result = await classifyWithEnvFallback(q, provider);
    return { status: 200, payload: result };
  } catch (e) {
    return {
      status: 502,
      payload: { error: e.message, keyFailed: isKeyFailureError(e.message), allFailed: true },
    };
  }
}
