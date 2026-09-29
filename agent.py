"""
agent.py
--------
The main entry point: a small AI AGENT that takes a natural-language user
request, uses a trained ML text classifier to figure out which TOOL the
user wants, and then calls that tool (module).

Tools available to the agent:
    - token_optimizer  : token counting / prompt compression / context management
    - image_generator  : autoencoder-based generative image tool
    - video_generator  : latent-space-interpolation generative video tool
    - chat (fallback)  : simple canned/echo response when no tool matches

Intent classification is done with a real (tiny) ML pipeline:
    TF-IDF vectorizer + Logistic Regression, trained on a small labeled
    set of example user utterances for each intent. This mirrors how
    production "router" or "orchestrator" agents decide which tool/function
    to call before invoking it (function-calling / tool-use pattern).

Usage:
    python agent.py                      # interactive CLI loop
    python agent.py "generate an image"  # single-shot mode
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import token_optimizer
import image_generator
import video_generator


# ---------------------------------------------------------------------
# 1. Training data for the intent classifier
# ---------------------------------------------------------------------
TRAINING_EXAMPLES = {
    "token": [
        "how many tokens does this text use",
        "optimize my prompt to save tokens",
        "compress this text for the LLM",
        "reduce token usage",
        "manage my chat context window",
        "cache repeated prompts",
        "count tokens in this sentence",
        "how do I save on token costs",
    ],
    "image": [
        "generate an image",
        "create a picture for me",
        "make some ai art",
        "draw a new image with your model",
        "produce a generated picture",
        "show me an ai generated shape",
        "can you create an image",
        "generate a synthetic picture",
    ],
    "video": [
        "generate a video",
        "make an animation",
        "create a short clip",
        "produce a generated video",
        "animate something for me",
        "make a video using ai",
        "render a video clip",
        "can you create a video",
    ],
    "chat": [
        "hello how are you",
        "what can you do",
        "tell me a joke",
        "who are you",
        "what is the weather like",
        "thanks for your help",
        "good morning",
        "explain what this project does",
    ],
}


def train_intent_classifier():
    texts, labels = [], []
    for label, examples in TRAINING_EXAMPLES.items():
        texts.extend(examples)
        labels.extend([label] * len(examples))

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)

    return vectorizer, clf


# ---------------------------------------------------------------------
# 2. The agent itself
# ---------------------------------------------------------------------
class Agent:
    def __init__(self):
        self.vectorizer, self.classifier = train_intent_classifier()
        self.last_intent = None
        self.last_result = None  # structured output of the last tool call
        self.memory = token_optimizer.SlidingWindowMemory(
            system_prompt="You are a helpful multi-tool AI agent.",
            max_tokens=300,
        )

    def classify_intent(self, user_text: str) -> str:
        X = self.vectorizer.transform([user_text])
        return self.classifier.predict(X)[0]

    def handle(self, user_text: str) -> str:
        self.memory.add("user", user_text)
        intent = self.classify_intent(user_text)
        self.last_intent, self.last_result = intent, None

        if intent == "token":
            summary = token_optimizer.demo()
            self.last_result = summary
            response = (
                f"[Tool: token_optimizer] Ran token-optimization demo. "
                f"Prompt compression saved {summary['savings_pct']:.0%} tokens; "
                f"cache hit rate was {summary['cache_hit_rate']:.0%}. "
                f"Full report saved to outputs/token_report.txt"
            )

        elif intent == "image":
            result = image_generator.generate_images(n_images=4)
            self.last_result = result
            response = (
                f"[Tool: image_generator] Trained a {result['latent_dim']}-d "
                f"latent autoencoder on {result['n_training_images']} synthetic "
                f"shapes (final MSE={result['training_loss']:.4f}) and generated "
                f"{len(result['generated_paths'])} new images -> "
                f"outputs/images/"
            )

        elif intent == "video":
            info = video_generator.render_video()
            self.last_result = info
            response = (
                f"[Tool: video_generator] Generated a {info['duration_sec']}s "
                f"video ({info['n_frames']} frames @ {info['fps']}fps) by "
                f"interpolating across {info['n_keyframes']} latent-space "
                f"keyframes -> {info['video_path']}"
            )

        else:
            response = (
                "[Tool: chat] I'm a demo agent that can optimize prompt "
                "tokens, generate images, or generate short videos. Try "
                "asking me to 'generate an image' or 'optimize my tokens'."
            )

        self.memory.add("assistant", response)
        return f"(intent detected: {intent})\n{response}"


# ---------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------
def main():
    agent = Agent()

    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
        print(f"> {user_text}")
        print(agent.handle(user_text))
        return

    print("AI Agent ready. Type a request (or 'quit' to exit).")
    print("Examples: 'generate an image', 'make a video', 'optimize my tokens'\n")
    while True:
        try:
            user_text = input("> ")
        except EOFError:
            break
        if user_text.strip().lower() in {"quit", "exit"}:
            break
        print(agent.handle(user_text), "\n")


if __name__ == "__main__":
    main()
