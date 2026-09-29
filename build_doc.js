const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ShadingType, ImageRun, BorderStyle, AlignmentType,
  LevelFormat, convertInchesToTwip
} = require("docx");
const fs = require("fs");

const CODE_FONT = "Consolas";
const BODY_FONT = "Calibri";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text, font: BODY_FONT, size: 22, ...opts })],
  });
}
function bullet(text) {
  return new Paragraph({
    text,
    bullet: { level: 0 },
    spacing: { after: 80 },
  });
}
function codeBlock(lines) {
  return new Paragraph({
    shading: { type: ShadingType.CLEAR, color: "auto", fill: "F0F0F0" },
    border: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
    },
    spacing: { after: 200, before: 100 },
    children: lines.split("\n").map((line, i) =>
      new TextRun({ text: line, font: CODE_FONT, size: 18, break: i === 0 ? 0 : 1 })
    ),
  });
}
function image(path, width, height) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200, before: 100 },
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(path),
        transformation: { width, height },
      }),
    ],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 260 },
    children: [new TextRun({ text, italics: true, size: 18, color: "555555", font: BODY_FONT })],
  });
}
function makeTable(headerRow, rows, widths) {
  const TOTAL_WIDTH = 9360; // US Letter usable width at 1" margins (12240 - 1440*2)
  const colWidths = widths || headerRow.map(() => Math.floor(TOTAL_WIDTH / headerRow.length));
  const mkCell = (text, idx, bold = false) =>
    new TableCell({
      width: { size: colWidths[idx], type: WidthType.DXA },
      shading: bold ? { type: ShadingType.CLEAR, color: "auto", fill: "2E5395" } : undefined,
      children: [new Paragraph({ children: [new TextRun({ text, bold, color: bold ? "FFFFFF" : "000000", size: 20, font: BODY_FONT })] })],
    });
  return new Table({
    width: { size: TOTAL_WIDTH, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headerRow.map((t, i) => mkCell(t, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((t, i) => mkCell(t, i)) })),
    ],
  });
}

const OUT = "/home/claude/ai_agent_project/outputs";

const doc = new Document({
  sections: [
    {
      properties: {
        page: { size: { width: 12240, height: 15840 } }, // US Letter
      },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
          children: [new TextRun({ text: "Multi-Tool AI Agent", bold: true, size: 48, font: BODY_FONT, color: "2E5395" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
          children: [new TextRun({ text: "Token Optimization • Image Generation • Video Generation", size: 26, italics: true, font: BODY_FONT, color: "555555" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
          children: [new TextRun({ text: "Project Documentation — Machine Learning / Deep Learning Coursework", size: 20, font: BODY_FONT, color: "888888" })],
        }),

        // ---------------- 1. Overview ----------------
        h1("1. Project Overview"),
        p("This project implements a small multi-tool AI agent that demonstrates three core concepts covered in class: token optimization for LLM-based chat systems, image generation using a deep-learning generative model, and video generation via latent-space interpolation. A lightweight NLP intent classifier acts as the \"agent\" layer, reading a user's natural-language request and routing it to the correct tool — the same function-calling / tool-use pattern used by production LLM agents."),
        p("All three tools are self-contained and run entirely offline: no pretrained weights or external APIs are downloaded. Training data (for the image/video model) is generated procedurally, and the neural network is trained from scratch inside the project."),

        h2("1.1 Objectives"),
        bullet("Show a practical understanding of token-level optimization techniques used when working with LLMs (counting, compression, context windows, caching)."),
        bullet("Build and train an actual neural network (autoencoder) that learns to generate new images, not just classify or reconstruct existing ones."),
        bullet("Extend the same trained model to generate short videos via latent-space interpolation."),
        bullet("Wrap all three tools behind a single conversational agent that uses ML (TF-IDF + Logistic Regression) to decide which tool to call."),

        h2("1.2 Project Structure"),
        codeBlock(
`ai_agent_project/
├── agent.py                    # main entry point (intent router)
├── modules/
│   ├── token_optimizer.py      # token counting + optimization
│   ├── image_generator.py      # autoencoder image generator
│   └── video_generator.py      # latent-space interpolation -> video
├── outputs/                    # generated images / video / reports
└── requirements.txt`
        ),

        // ---------------- 2. Architecture ----------------
        h1("2. System Architecture"),
        p("The agent receives a natural-language request, vectorizes it with TF-IDF, classifies the intent with Logistic Regression into one of four classes (token / image / video / chat), and dispatches to the matching module. Each module returns a structured result, which the agent turns into a natural-language response and stores in a sliding-window conversation memory."),
        makeTable(
          ["Component", "Technique", "Purpose"],
          [
            ["Intent classifier", "TF-IDF + Logistic Regression", "Route free-text requests to the correct tool"],
            ["token_optimizer", "Tokenization, compression, caching", "Reduce LLM token spend / manage context window"],
            ["image_generator", "MLP Autoencoder (feed-forward NN)", "Generate new images from a learned latent space"],
            ["video_generator", "Latent-space linear interpolation", "Generate video frames from the trained decoder"],
            ["Conversation memory", "Sliding-window FIFO trimming", "Keep chat history under a token budget"],
          ],
          [2200, 2860, 4300]
        ),

        // ---------------- 3. Token optimization ----------------
        h1("3. Module 1 — Token Optimization"),
        p("This module demonstrates four practical techniques for controlling LLM token usage in a chat agent."),
        h2("3.1 Techniques implemented"),
        bullet("Token counting — a BPE-style tokenizer estimates how many tokens a string will cost (a local regex-based tokenizer is used as a stand-in for tiktoken, since this sandbox has no internet access to OpenAI's hosted vocabulary files)."),
        bullet("Prompt compression — strips filler / stop words from a prompt before sending it to the model, when exact grammar doesn't matter (e.g. system instructions)."),
        bullet("Sliding-window context memory — keeps a running chat history under a fixed token budget by dropping the oldest turns first, always preserving the system prompt."),
        bullet("Response caching — hashes incoming prompts (case/whitespace-normalized) so identical or near-identical questions are never recomputed."),

        h2("3.2 Key code — sliding window memory"),
        codeBlock(
`class SlidingWindowMemory:
    def __init__(self, system_prompt, max_tokens=200):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.turns = []

    def add(self, role, text):
        self.turns.append((role, text))
        self._trim()

    def _trim(self):
        while self._total_tokens() > self.max_tokens and len(self.turns) > 1:
            self.turns.pop(0)   # drop oldest turn first`
        ),

        h2("3.3 Results"),
        makeTable(
          ["Metric", "Value"],
          [
            ["Original prompt tokens", "36"],
            ["Compressed prompt tokens", "27"],
            ["Token savings from compression", "25%"],
            ["Cache hit rate (demo run)", "50%"],
            ["Tokens saved via caching", "~18 tokens"],
          ]
        ),
        p("In the demo conversation, the sliding-window memory kept the context under a 60-token budget by automatically dropping the two oldest turns once the budget was exceeded, while always preserving the system prompt — a simplified version of how real chat agents manage limited context windows."),

        // ---------------- 4. Image generation ----------------
        h1("4. Module 2 — Image Generation (Deep Learning)"),
        p("The image generator is a genuine feed-forward neural network trained as an autoencoder: input (flattened 32×32 image, 1024 values) → encoder → 8-dimensional latent bottleneck → decoder → output (1024 values). It is trained with scikit-learn's MLPRegressor using backpropagation and the Adam optimizer, with the reconstruction target equal to the input."),

        h2("4.1 Training data"),
        p("Since no internet access is available to download an image dataset, training data is generated procedurally: 400 grayscale 32×32 images of circles, squares and triangles at randomized positions and sizes."),
        image(`${OUT}/images/training_samples.png`, 480, 80),
        caption("Figure 1 — Sample of the procedurally generated training images (circles, triangles, squares)."),

        h2("4.2 How generation works"),
        p("A trained autoencoder can only reconstruct inputs it is given. To actually generate new images, this project splits the network's own learned weight matrices at the bottleneck layer and runs only the decoder half (bottleneck → output) on brand-new, randomly sampled latent vectors drawn from the distribution of latents the encoder produces. This mirrors — in simplified form — how a GAN or VAE decoder generates unseen images from random latent noise."),
        codeBlock(
`# Layer topology for hidden_layer_sizes=(128, LATENT_DIM, 128):
#   input(1024) -> h1(128) -> bottleneck(8) -> h2(128) -> output(1024)
decoder_weights = model.coefs_[2:]       # bottleneck -> output only
decoder_biases  = model.intercepts_[2:]

def decode(latent_vec, weights, biases):
    a = latent_vec
    for i, (W, b) in enumerate(zip(weights, biases)):
        a = a @ W + b
        if i < len(weights) - 1:
            a = relu(a)
    return np.clip(a, 0, 1)`
        ),

        h2("4.3 Results"),
        makeTable(
          ["Metric", "Value"],
          [
            ["Training images", "400 (procedurally generated)"],
            ["Architecture", "1024 → 128 → 8 → 128 → 1024"],
            ["Latent dimension", "8"],
            ["Final training MSE", "0.0278"],
            ["Images generated per run", "4–6"],
          ]
        ),
        image(`${OUT}/images/generated_grid.png`, 480, 80),
        caption("Figure 2 — New images generated by decoding random latent vectors (never seen during training)."),
        p("The generated outputs are soft, blob-like versions of the training shapes rather than crisp geometric figures — an expected result for a small, fully-connected (non-convolutional) autoencoder with only an 8-dimensional latent space. Using a convolutional architecture and/or a larger latent space would sharpen results, at the cost of more training data and compute."),

        // ---------------- 5. Video generation ----------------
        h1("5. Module 3 — Video Generation (Latent Interpolation)"),
        p("The video generator reuses the decoder trained in Module 2. It samples several random keyframe vectors in latent space, linearly interpolates between consecutive keyframes to produce a smooth path, and decodes every point along that path into a video frame. The frames are then written to an .mp4 file with OpenCV."),
        h2("5.1 Key code"),
        codeBlock(
`for i in range(len(keyframes) - 1):
    z_start, z_end = keyframes[i], keyframes[i + 1]
    for step in range(frames_per_transition):
        t = step / frames_per_transition
        z = lerp(z_start, z_end, t)              # interpolate in latent space
        frame = decode(z, decoder_weights, decoder_biases)
        writer.write(upscale(frame))`
        ),
        h2("5.2 Results"),
        makeTable(
          ["Metric", "Value"],
          [
            ["Output file", "outputs/generated_video.mp4"],
            ["Resolution", "256 × 256 (upscaled from 32×32 frames)"],
            ["Frame rate", "12 fps"],
            ["Total frames", "60"],
            ["Duration", "5.0 seconds"],
            ["Latent keyframes / transitions", "4"],
          ]
        ),
        image(`${OUT}/video_midframe.png`, 220, 220),
        caption("Figure 3 — A mid-transition frame extracted from the generated video, showing a smooth morph between shapes."),

        // ---------------- 6. Agent ----------------
        h1("6. The Agent — Tool Routing with ML"),
        p("agent.py ties the three modules together behind a single conversational interface. A TF-IDF vectorizer converts the user's free-text request into a numeric feature vector, and a Logistic Regression classifier — trained on a small labeled set of example phrases per intent — predicts which tool to invoke."),
        h2("6.1 Example runs"),
        makeTable(
          ["User input", "Detected intent", "Tool invoked"],
          [
            ["\"can you generate an image for me\"", "image", "image_generator"],
            ["\"how do I save tokens on my prompt\"", "token", "token_optimizer"],
            ["\"make me a short animation\"", "video", "video_generator"],
            ["\"hello there, what can you do\"", "chat", "fallback response"],
          ],
          [4160, 2200, 3000]
        ),
        p("This mirrors the \"function calling\" / \"tool use\" pattern found in modern LLM agent frameworks: an ML model decides which function/tool to call based on the user's message, and that tool's structured output is turned back into natural language for the user."),

        // ---------------- 7. Conclusion ----------------
        h1("7. Limitations & Future Work"),
        bullet("The tokenizer is a local approximation, not the real tiktoken BPE vocabulary (blocked by sandbox network restrictions) — token counts will differ slightly from an actual OpenAI model."),
        bullet("The image/video generator uses a small fully-connected autoencoder rather than a convolutional GAN/diffusion model, so outputs are low-resolution and soft-edged."),
        bullet("The intent classifier is trained on a small hand-written dataset (8 examples per class); a production agent would use a larger, more diverse training set or an LLM-based classifier."),
        bullet("Future work: swap in a convolutional autoencoder or small diffusion model for sharper image/video generation, and connect the token optimizer to a real LLM API call."),

        h1("8. Conclusion"),
        p("This project demonstrates, end-to-end, three core ML/DL concepts from this week's coursework — token optimization, generative image modeling, and generative video via latent interpolation — unified behind a single ML-routed agent. Every component (the autoencoder, the intent classifier, and the token-optimization utilities) is trained or computed from scratch inside this project, with no external pretrained models or internet-downloaded datasets required."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("/home/claude/ai_agent_project/outputs/AI_Agent_Documentation.docx", buffer);
  console.log("Document written.");
});
