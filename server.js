
const express = require("express");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json({ limit: "2mb" }));

app.get("/", (req, res) => {
  res.json({
    name: "VORTEX AI Server",
    status: "online",
    ai: "Cloudflare Workers AI"
  });
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;

    if (!message || typeof message !== "string") {
      return res.status(400).json({
        error: "Message is required"
      });
    }

    if (!process.env.CLOUDFLARE_ACCOUNT_ID ||
        !process.env.CLOUDFLARE_API_TOKEN) {
      return res.status(500).json({
        error: "Cloudflare credentials are missing"
      });
    }

    const response = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct`,
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.CLOUDFLARE_API_TOKEN}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          messages: [
            {
              role: "system",
              content: "You are VORTEX AI, a helpful assistant."
            },
            {
              role: "user",
              content: message
            }
          ],
          max_tokens: 512
        })
      }
    );

    const data = await response.json();

    if (!response.ok || !data.success) {
      return res.status(response.status || 500).json({
        error: "Cloudflare AI request failed",
        details: data.errors || data
      });
    }

    res.json({
      success: true,
      response: data.result?.response || data.result
    });

  } catch (error) {
    res.status(500).json({
      error: "Server error",
      details: error.message
    });
  }
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`VORTEX server running on port ${PORT}`);
});
