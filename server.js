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



// Experimental local text-to-motion engine (runs on this Render service; no external inference API).
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");
const { execFile } = require("node:child_process");
const { promisify } = require("node:util");
const { pipeline } = require("node:stream/promises");
const { Readable } = require("node:stream");
const execFileAsync = promisify(execFile);

const MOTION_ROOT = path.join(os.tmpdir(), "vortex-sprongmotion");
const MOTION_MODEL = path.join(MOTION_ROOT, "SprongMotion-14M-v1.0-Q8_0.gguf");
const MOTION_BIN = path.join(MOTION_ROOT, "sprongmotion-generate");
const MOTION_SHA256 = "68fbd311a0d7d75f60508b35254171583c85880e7e124140457b57ac84c5de9d";
let motionInitPromise = null;
let motionBusy = false;

const MOTION_HELPER_SOURCE = String.raw`
#include <stdio.h>
#include <stdlib.h>
#include "sprongmotion.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: sprongmotion-generate model.gguf prompt seconds\\n");
        return 2;
    }

    char *end = NULL;
    float seconds = strtof(argv[3], &end);
    if (end == argv[3] || *end != '\\0' || seconds < 0.5f || seconds > 3.0f) {
        fprintf(stderr, "seconds must be between 0.5 and 3.0\\n");
        return 2;
    }

    sm_set_thread_count(1);
    sm_model model = sm_load_model(argv[1]);
    if (!sm_is_model_valid(model)) {
        fprintf(stderr, "%s\\n", sm_get_error());
        return 1;
    }

    sm_options options = sm_default_options(model);
    options.seconds = seconds;
    options.steps = 4;
    options.residual_layers = 1;
    options.foot_lock = true;

    sm_motion motion = sm_generate_motion_ex(model, argv[2], options);
    if (!sm_is_motion_valid(motion)) {
        fprintf(stderr, "%s\\n", sm_get_error());
        sm_unload_model(model);
        return 1;
    }

    char *bvh = sm_export_motion_bvh_text(motion, 100.0f);
    if (!bvh) {
        fprintf(stderr, "%s\\n", sm_get_error());
        sm_unload_motion(motion);
        sm_unload_model(model);
        return 1;
    }

    fputs(bvh, stdout);
    sm_free(bvh);
    sm_unload_motion(motion);
    sm_unload_model(model);
    return 0;
}
`;

async function downloadMotionFile(url, destination) {
  const response = await fetch(url, { signal: AbortSignal.timeout(180000) });
  if (!response.ok || !response.body) {
    throw new Error("Download failed with HTTP " + response.status);
  }
  await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(destination));
}

async function ensureMotionRuntime() {
  if (motionInitPromise) return motionInitPromise;
  motionInitPromise = (async () => {
    await fsp.mkdir(MOTION_ROOT, { recursive: true });
    const sourceDir = path.join(MOTION_ROOT, "sprongmotion-runtime-main");
    const helperPath = path.join(MOTION_ROOT, "generate.c");

    if (!fs.existsSync(MOTION_BIN)) {
      const archive = path.join(MOTION_ROOT, "runtime.tar.gz");
      if (!fs.existsSync(path.join(sourceDir, "Makefile"))) {
        await downloadMotionFile(
          "https://codeload.github.com/SegerEnd/sprongmotion-runtime/tar.gz/refs/heads/main",
          archive
        );
        await execFileAsync("tar", ["-xzf", archive, "-C", MOTION_ROOT], { timeout: 60000 });
        await fsp.rm(archive, { force: true });
      }

      await execFileAsync("make", ["lib"], { cwd: sourceDir, timeout: 180000, maxBuffer: 4 * 1024 * 1024 });
      await fsp.writeFile(helperPath, MOTION_HELPER_SOURCE, "utf8");
      await execFileAsync("cc", [
        "-std=c11", "-O2", "-I", path.join(sourceDir, "include"),
        helperPath, path.join(sourceDir, "build", "libsprongmotion.a"),
        "-lm", "-pthread", "-o", MOTION_BIN
      ], { timeout: 120000, maxBuffer: 4 * 1024 * 1024 });
    }

    if (!fs.existsSync(MOTION_MODEL)) {
      const partial = MOTION_MODEL + ".part";
      await downloadMotionFile(
        "https://huggingface.co/SegerEnd/SprongMotion-14M/resolve/main/SprongMotion-14M-v1.0-Q8_0.gguf?download=true",
        partial
      );
      const digest = crypto.createHash("sha256").update(await fsp.readFile(partial)).digest("hex");
      if (digest !== MOTION_SHA256) {
        await fsp.rm(partial, { force: true });
        throw new Error("SprongMotion model checksum mismatch");
      }
      await fsp.rename(partial, MOTION_MODEL);
    }

    return true;
  })().catch(error => {
    motionInitPromise = null;
    throw error;
  });
  return motionInitPromise;
}

// Build a short skeletal animation from an English action prompt.
// The model and C runtime execute on this Render instance, not on a remote inference provider.
app.post("/api/motion", async (req, res) => {
  const prompt = req.body?.prompt;
  const seconds = Number(req.body?.seconds ?? 1.5);
  if (typeof prompt !== "string" || !prompt.trim() || prompt.length > 240) {
    return res.status(400).json({ success: false, error: "Prompt must contain 1 to 240 characters." });
  }
  if (!Number.isFinite(seconds) || seconds < 0.5 || seconds > 3) {
    return res.status(400).json({ success: false, error: "Duration must be between 0.5 and 3 seconds." });
  }
  if (motionBusy) {
    return res.status(429).json({ success: false, error: "The local motion engine is busy. Try again shortly." });
  }

  motionBusy = true;
  try {
    await ensureMotionRuntime();
    const { stdout } = await execFileAsync(
      MOTION_BIN,
      [MOTION_MODEL, prompt.trim(), String(seconds)],
      { timeout: 120000, maxBuffer: 12 * 1024 * 1024 }
    );
    if (!stdout || !stdout.startsWith("HIERARCHY")) {
      return res.status(502).json({ success: false, error: "The motion model returned an invalid BVH file." });
    }
    res.setHeader("Content-Type", "application/octet-stream");
    res.setHeader("Content-Disposition", 'attachment; filename="vortex-motion.bvh"');
    res.setHeader("X-Vortex-Model", "SprongMotion-14M");
    return res.send(stdout);
  } catch (error) {
    console.error("VORTEX local motion engine error:", error.message);
    return res.status(503).json({
      success: false,
      error: "The local motion engine could not start or finish on this server.",
      details: error.message
    });
  } finally {
    motionBusy = false;
  }
});

app.get("/api/motion/status", (req, res) => {
  res.json({
    success: true,
    engine: "SprongMotion-14M",
    execution: "local-render-server",
    ready: fs.existsSync(MOTION_BIN) && fs.existsSync(MOTION_MODEL),
    output: "BVH skeletal animation",
    maxSeconds: 3
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`VORTEX server running on port ${PORT}`);
});
