
const express = require("express");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json({ limit: "2mb" }));

// Server status
app.get("/", (req, res) => {
  res.json({
    name: "VORTEX AI Server",
    status: "online",
    ai: "Cloudflare Workers AI",
    model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
  });
});

// Health check
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    server: "VORTEX"
  });
});

// AI Chat API
app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body || {};

    if (!message || typeof message !== "string" || !message.trim()) {
      return res.status(400).json({
        success: false,
        error: "A valid message is required"
      });
    }

    if (
      !process.env.CLOUDFLARE_ACCOUNT_ID ||
      !process.env.CLOUDFLARE_API_TOKEN
    ) {
      return res.status(500).json({
        success: false,
        error: "Cloudflare credentials are missing"
      });
    }

    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    const apiToken = process.env.CLOUDFLARE_API_TOKEN;

    const apiUrl =
      `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast`;

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiToken}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        messages: [
          {
            role: "system",
            content: "You are VORTEX AI, a helpful and intelligent assistant. Respond clearly in the user's language."
          },
          {
            role: "user",
            content: message.trim()
          }
        ],
        max_tokens: 512
      })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      console.error("Cloudflare API error:", data.errors || data);

      return res.status(response.status || 502).json({
        success: false,
        error: "Cloudflare AI request failed",
        details: data.errors || data
      });
    }

    return res.json({
      success: true,
      model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
      response: data.result?.response || ""
    });

  } catch (error) {
    console.error("VORTEX server error:", error);

    return res.status(500).json({
      success: false,
      error: "Internal server error",
      details: error.message
    });
  }
});

// Start server
const PORT = process.env.PORT || 3000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`VORTEX server running on port ${PORT}`);
});
