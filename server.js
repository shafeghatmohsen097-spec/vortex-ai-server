const express = require("express");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json({ limit: "2mb" }));

const IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell";

// Server status
app.get("/", (req, res) => {
  res.json({
    name: "VORTEX AI Server",
    status: "online",
    ai: "Cloudflare Workers AI",
    model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    imageModel: IMAGE_MODEL
  });
});

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", server: "VORTEX" });
});

function cloudflareCredentials() {
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
  const apiToken = process.env.CLOUDFLARE_API_TOKEN;
  if (!accountId || !apiToken) return null;
  return { accountId, apiToken };
}

// AI Chat API
app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body || {};
    if (!message || typeof message !== "string" || !message.trim()) {
      return res.status(400).json({ success: false, error: "A valid message is required" });
    }
    const credentials = cloudflareCredentials();
    if (!credentials) {
      return res.status(500).json({ success: false, error: "Cloudflare credentials are missing" });
    }
    const apiUrl = `https://api.cloudflare.com/client/v4/accounts/${credentials.accountId}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast`;
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${credentials.apiToken}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        messages: [
          { role: "system", content: "You are VORTEX AI, a helpful and intelligent assistant. Respond clearly in the user's language." },
          { role: "user", content: message.trim() }
        ],
        max_tokens: 512
      }),
      signal: AbortSignal.timeout(90000)
    });
    const data = await response.json();
    if (!response.ok || !data.success) {
      console.error("Cloudflare chat request failed:", response.status);
      return res.status(response.status || 502).json({
        success: false,
        error: "Cloudflare AI request failed",
        details: data.errors || undefined
      });
    }
    return res.json({
      success: true,
      model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
      response: data.result?.response || ""
    });
  } catch (error) {
    console.error("VORTEX chat error:", error.message);
    return res.status(500).json({ success: false, error: "Internal server error" });
  }
});

// Image generation API - uses the existing Cloudflare token stored on Render.
app.post("/api/image", async (req, res) => {
  try {
    const prompt = req.body?.prompt;
    if (typeof prompt !== "string" || !prompt.trim() || prompt.length > 2048) {
      return res.status(400).json({ success: false, error: "Prompt must contain 1 to 2048 characters." });
    }
    const credentials = cloudflareCredentials();
    if (!credentials) {
      return res.status(500).json({ success: false, error: "Cloudflare credentials are missing on the server." });
    }
    const endpoint = `https://api.cloudflare.com/client/v4/accounts/${credentials.accountId}/ai/run/${IMAGE_MODEL}`;
    const upstream = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${credentials.apiToken}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ prompt: prompt.trim(), steps: 4 }),
      signal: AbortSignal.timeout(120000)
    });
    const data = await upstream.json().catch(() => null);
    if (!upstream.ok || !data?.success) {
      console.error("Cloudflare image request failed:", upstream.status, data?.errors || "");
      return res.status(upstream.status || 502).json({
        success: false,
        error: "Cloudflare image generation failed.",
        details: data?.errors || undefined
      });
    }
    const image = data.result?.image;
    if (typeof image !== "string" || !image.length) {
      return res.status(502).json({ success: false, error: "Image model returned no image." });
    }
    return res.json({
      success: true,
      image: `data:image/jpeg;base64,${image}`,
      model: IMAGE_MODEL
    });
  } catch (error) {
    console.error("VORTEX image error:", error.message);
    const timeout = error.name === "TimeoutError" || error.name === "AbortError";
    return res.status(timeout ? 504 : 502).json({
      success: false,
      error: timeout ? "Image generation timed out. Please try again." : "Could not reach the image generation service."
    });
  }
});

// Persian text-to-speech through Cloudflare's unified AI catalog.
// Uses the existing server-side Cloudflare token; audio generation is metered.
app.post("/api/speech", async (req, res) => {
  try {
    const text = req.body?.text;
    if (typeof text !== "string" || !text.trim() || text.length > 3500) {
      return res.status(400).json({ success: false, error: "Text must contain 1 to 3500 characters." });
    }
    const credentials = cloudflareCredentials();
    if (!credentials) {
      return res.status(500).json({ success: false, error: "Cloudflare credentials are missing on the server." });
    }

    const endpoint = `https://api.cloudflare.com/client/v4/accounts/${credentials.accountId}/ai/run`;
    const upstream = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${credentials.apiToken}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "openai/tts-1",
        input: {
          text: text.trim(),
          voice: "nova",
          response_format: "mp3",
          speed: 1
        }
      }),
      signal: AbortSignal.timeout(120000)
    });

    const data = await upstream.json().catch(() => null);
    if (!upstream.ok || !data?.success) {
      console.error("Cloudflare speech request failed:", upstream.status, data?.errors || "");
      return res.status(upstream.status || 502).json({
        success: false,
        error: "Cloudflare speech generation failed.",
        details: data?.errors || undefined
      });
    }

    const audioRef = data.result?.audio;
    if (typeof audioRef !== "string" || !audioRef) {
      return res.status(502).json({ success: false, error: "Speech model returned no audio file." });
    }

    let audioBuffer;
    if (audioRef.startsWith("data:")) {
      const encoded = audioRef.split(",")[1] || "";
      audioBuffer = Buffer.from(encoded, "base64");
    } else if (/^https:\/\//i.test(audioRef)) {
      const audioResponse = await fetch(audioRef, { signal: AbortSignal.timeout(60000) });
      if (!audioResponse.ok) {
        return res.status(502).json({ success: false, error: "Could not download the generated audio." });
      }
      audioBuffer = Buffer.from(await audioResponse.arrayBuffer());
    } else {
      audioBuffer = Buffer.from(audioRef, "base64");
    }

    if (!audioBuffer.length) {
      return res.status(502).json({ success: false, error: "Generated audio file was empty." });
    }

    return res.json({
      success: true,
      audio: audioBuffer.toString("base64"),
      mimeType: "audio/mpeg",
      model: "openai/tts-1 via Cloudflare AI"
    });
  } catch (error) {
    console.error("VORTEX speech error:", error.message);
    const timeout = error.name === "TimeoutError" || error.name === "AbortError";
    return res.status(timeout ? 504 : 502).json({
      success: false,
      error: timeout ? "Speech generation timed out. Please try again." : "Could not reach the speech generation service."
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`VORTEX server running on port ${PORT}`);
});
