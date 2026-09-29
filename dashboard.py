"""
dashboard.py
------------
A local web dashboard for the multi-tool AI agent project, built with
Streamlit.

Run with:
    streamlit run dashboard.py
"""

import sys
import os
import json
import glob

import streamlit as st

# ---------------------------------------------------------------------
# Make modules/ available for imports
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE_DIR, "modules")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

if MODULES_DIR not in sys.path:
    sys.path.append(MODULES_DIR)

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import token_optimizer
import image_generator
import video_generator

from agent import Agent

# ---------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="AI Agent Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Tool AI Agent Dashboard")

st.caption(
    "Token optimization • Image generation • Video generation — "
    "all running locally, no internet required."
)

# ---------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------

# Response cache
if "token_cache" not in st.session_state:
    st.session_state.token_cache = token_optimizer.ResponseCache()

# Agent
if "agent" not in st.session_state:
    with st.spinner("Starting agent (training intent classifier)..."):
        st.session_state.agent = Agent()

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tab_chat, tab_token, tab_image, tab_video = st.tabs(
    [
        "💬 Agent Chat",
        "🔢 Token Optimizer",
        "🖼️ Image Generator",
        "🎬 Video Generator"
    ]
)

# =====================================================================
# TAB 1: AGENT CHAT
# =====================================================================

with tab_chat:

    st.subheader("Talk to the agent")

    st.write(
        "Type a request in plain English — the agent will classify "
        "your intent and call the matching tool automatically."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def show_artifacts(intent, data):
        """Render tool output inside the chat."""

        if not data:
            return

        if intent == "image":

            if "generated_paths" in data:
                st.image(
                    data["generated_paths"],
                    width=150
                )

        elif intent == "video":

            if "video_path" in data:
                st.video(data["video_path"])

        elif intent == "token":

            if "raw_tokens" in data:

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Original tokens",
                    data["raw_tokens"]
                )

                c2.metric(
                    "Compressed tokens",
                    data["compressed_tokens"]
                )

                c3.metric(
                    "Cache hit rate",
                    f"{data['cache_hit_rate']:.0%}"
                )

    # Display previous messages
    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):

            st.write(msg["text"])

            show_artifacts(
                msg.get("intent"),
                msg.get("data")
            )

    # Chat input
    user_text = st.chat_input(
        "e.g. 'generate an image' or 'optimize my tokens'"
    )

    if user_text:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "text": user_text
            }
        )

        with st.spinner("Agent is working..."):

            response = st.session_state.agent.handle(
                user_text
            )

        agent = st.session_state.agent

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "text": response,
                "intent": agent.last_intent,
                "data": agent.last_result
            }
        )

        st.rerun()


# =====================================================================
# TAB 2: TOKEN OPTIMIZER
# =====================================================================

with tab_token:

    st.subheader("🔢 Token Optimization Demo")

    st.write(
        "Enter your own prompt to see token counting, "
        "prompt compression, and token savings."
    )

    # -----------------------------------------------------------------
    # Prompt optimization
    # -----------------------------------------------------------------

    user_prompt = st.text_area(
        "Enter your prompt:",
        placeholder=(
            "Example: Can you please explain machine learning "
            "in very simple terms?"
        ),
        height=120
    )

    if st.button(
        "▶ Optimize Prompt",
        key="optimize_prompt"
    ):

        if not user_prompt.strip():

            st.warning(
                "Please enter a prompt first."
            )

        else:

            with st.spinner("Optimizing prompt..."):

                result = token_optimizer.optimize_prompt(
                    user_prompt
                )

            st.session_state.user_token_result = result

    # -----------------------------------------------------------------
    # Display optimization result
    # -----------------------------------------------------------------

    if "user_token_result" in st.session_state:

        result = st.session_state.user_token_result

        st.divider()

        st.subheader("Optimization Result")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Original Tokens",
            result["raw_tokens"]
        )

        c2.metric(
            "Compressed Tokens",
            result["compressed_tokens"]
        )

        c3.metric(
            "Token Savings",
            f"{result['savings_pct']:.0%}"
        )

        st.write("### Original Prompt")

        st.info(
            result["original_prompt"]
        )

        st.write("### Compressed Prompt")

        st.success(
            result["compressed_prompt"]
        )

        st.write(
            f"**Tokens saved:** "
            f"{result['tokens_saved']}"
        )

    # -----------------------------------------------------------------
    # Response caching
    # -----------------------------------------------------------------

    st.divider()

    st.subheader("Response Caching")

    st.write(
        "Enter the same question twice to demonstrate "
        "CACHE MISS followed by CACHE HIT."
    )

    cache_prompt = st.text_input(
        "Enter a question to test caching:",
        placeholder="What is machine learning?",
        key="cache_prompt"
    )

    if st.button(
        "Check Cache",
        key="check_cache"
    ):

        if not cache_prompt.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            cache = st.session_state.token_cache

            def fake_llm_call(prompt):
                return (
                    f"Generated answer for: {prompt}"
                )

            result, was_cached = cache.get_or_compute(
                cache_prompt,
                fake_llm_call
            )

            if was_cached:

                st.success(
                    "CACHE HIT — previous result reused."
                )

            else:

                st.info(
                    "CACHE MISS — answer had to be computed."
                )

            st.write("### Result")

            st.write(result)

            c1, c2 = st.columns(2)

            c1.metric(
                "Cache Hits",
                cache.hits
            )

            c2.metric(
                "Cache Misses",
                cache.misses
            )

    # -----------------------------------------------------------------
    # Existing token summary
    # -----------------------------------------------------------------

    summary_path = os.path.join(
        OUT_DIR,
        "token_summary.json"
    )

    if (
        "token_summary" not in st.session_state
        and os.path.exists(summary_path)
    ):

        with open(summary_path) as f:

            st.session_state.token_summary = json.load(f)

    if "token_summary" in st.session_state:

        s = st.session_state.token_summary

        st.divider()

        st.subheader("Previous Demo Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Original tokens",
            s["raw_tokens"]
        )

        c2.metric(
            "Compressed tokens",
            s["compressed_tokens"]
        )

        c3.metric(
            "Token savings",
            f"{s['savings_pct']:.0%}"
        )

        c4.metric(
            "Cache hit rate",
            f"{s['cache_hit_rate']:.0%}"
        )


# =====================================================================
# TAB 3: IMAGE GENERATOR
# =====================================================================

with tab_image:

    st.subheader("🖼️ AI Image Generation")

    st.write(
        "Trains a small autoencoder on synthetic shapes, "
        "then generates new images by decoding random latent vectors."
    )

    n_images = st.slider(
        "Number of images to generate",
        2,
        8,
        4
    )

    if st.button(
        "▶ Train & generate images",
        key="run_image"
    ):

        with st.spinner(
            "Training autoencoder and generating images..."
        ):

            result = image_generator.generate_images(
                n_images=n_images
            )

        st.session_state.image_result = {
            "loss": result["training_loss"],
            "n_train": result["n_training_images"],
            "latent_dim": result["latent_dim"],
            "generated_paths": result["generated_paths"],
            "training_sample_path": result[
                "training_sample_path"
            ]
        }

    if "image_result" in st.session_state:

        r = st.session_state.image_result

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Training images",
            r["n_train"]
        )

        c2.metric(
            "Latent dimension",
            r["latent_dim"]
        )

        c3.metric(
            "Final MSE loss",
            f"{r['loss']:.4f}"
        )

        st.write(
            "**Training samples (real shapes):**"
        )

        st.image(
            r["training_sample_path"]
        )

        st.write(
            "**Generated images "
            "(new, never seen during training):**"
        )

        cols = st.columns(
            len(r["generated_paths"])
        )

        for col, path in zip(
            cols,
            r["generated_paths"]
        ):

            col.image(path)

    else:

        existing = sorted(
            glob.glob(
                os.path.join(
                    OUT_DIR,
                    "images",
                    "generated_*.png"
                )
            )
        )

        if existing:

            st.write(
                "**Previously generated images:**"
            )

            cols = st.columns(
                len(existing)
            )

            for col, path in zip(
                cols,
                existing
            ):

                col.image(path)

        else:

            st.info(
                "Click the button above to train "
                "the model and generate images."
            )


# =====================================================================
# TAB 4: VIDEO GENERATOR
# =====================================================================

with tab_video:

    st.subheader("🎬 AI Video Generation")

    st.write(
        "Reuses the trained image decoder and generates "
        "a short video by interpolating through latent space."
    )

    colorful = st.checkbox(
        "Colourful mode 🌈",
        value=True
    )

    if st.button(
        "▶ Generate video",
        key="run_video"
    ):

        with st.spinner(
            "Training decoder and rendering video frames..."
        ):

            info = video_generator.render_video(
                colorful=colorful
            )

        st.session_state.video_info = info

    if "video_info" in st.session_state:

        info = st.session_state.video_info

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Frames",
            info["n_frames"]
        )

        c2.metric(
            "FPS",
            info["fps"]
        )

        c3.metric(
            "Duration",
            f"{info['duration_sec']}s"
        )

        st.video(
            info["video_path"]
        )

    elif os.path.exists(
        os.path.join(
            OUT_DIR,
            "generated_video.mp4"
        )
    ):

        st.write(
            "**Previously generated video:**"
        )

        st.video(
            os.path.join(
                OUT_DIR,
                "generated_video.mp4"
            )
        )

    else:

        st.info(
            "Click the button above to generate a video."
        )


# =====================================================================
# FOOTER
# =====================================================================

st.divider()

st.caption(
    "All models are trained from scratch on procedurally generated "
    "data — no internet or pretrained weights used."
)