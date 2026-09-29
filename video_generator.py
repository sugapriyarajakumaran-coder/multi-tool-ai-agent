"""
video_generator.py
-------------------
Generates a short video by walking smoothly through the LATENT SPACE of
the autoencoder trained in image_generator.py and decoding each point
along the path into a frame. This is a simplified version of the same
core idea used in real generative-video systems: a sequence of frames is
produced by moving through a learned latent space rather than being
hand-drawn or filmed.

Pipeline:
    1. Reuse the trained decoder from image_generator.py
    2. Pick two (or more) random latent vectors as keyframes
    3. Linearly interpolate between them to get N in-between latent vectors
    4. Decode every interpolated vector into an image frame
    5. Stitch the frames into an .mp4 with OpenCV's VideoWriter
"""

import os
import numpy as np
import cv2

from image_generator import generate_images, decode, IMG_SIZE

OUT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs"
    )
)
FRAME_SIZE = 256  # upscaled for visibility
FPS = 12


def lerp(a, b, t):
    return a * (1 - t) + b * t


def colorize(gray, t):
    """
    Turn a grayscale generated frame (H x W, values 0-1) into a colourful
    RGB frame. The neural network decides the SHAPE (brightness); this
    step adds a "style" on top: a rainbow gradient whose hue keeps
    rotating over time (t goes 0 -> 1 across the whole video).
    """
    h, w = gray.shape
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    # foreground: rainbow across the frame, drifting with time and shape
    hue_fg = (xx * 0.6 + yy * 0.3 + t * 2.0 + gray * 0.25) % 1.0
    fg = np.stack([hue_fg * 179, np.full_like(gray, 230), np.full_like(gray, 255)], axis=-1)

    # background: dark, softly glowing gradient in the opposite hue
    hue_bg = (t * 2.0 + 0.5 + yy * 0.4) % 1.0
    bg = np.stack([hue_bg * 179, np.full_like(gray, 200), 40 + 60 * (1 - yy)], axis=-1)

    # blend: shape (gray) reveals the bright rainbow over the dark background
    a = np.clip(gray, 0, 1)[..., None]
    hsv = (bg * (1 - a) + fg * a).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def build_latent_path(lat_mean, lat_std, n_keyframes=4, seed=7):
    """Random keyframes in latent space, to be interpolated between."""
    rng = np.random.default_rng(seed)
    keyframes = []
    for _ in range(n_keyframes):
        z = lat_mean + rng.normal(size=len(lat_mean)) * lat_std * 1.1
        z = np.maximum(z, 0)  # relu-consistent latent space
        keyframes.append(z)
    keyframes.append(keyframes[0])  # loop back to start for a seamless loop
    return keyframes


def render_video(frames_per_transition=15, seed=7, colorful=True):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Train (or reuse) the autoencoder + get decoder weights / latent stats
    gen_result = generate_images(n_images=1, seed=seed)
    decoder_weights = gen_result["decoder_weights"]
    decoder_biases = gen_result["decoder_biases"]
    lat_mean, lat_std = gen_result["latent_stats"]

    keyframes = build_latent_path(lat_mean, lat_std, n_keyframes=4, seed=seed)

    video_path = os.path.join(OUT_DIR, "generated_video.mp4")

    # 1) build all frames first
    frames = []
    total = (len(keyframes) - 1) * frames_per_transition
    for i in range(len(keyframes) - 1):
        z_start, z_end = keyframes[i], keyframes[i + 1]
        for step in range(frames_per_transition):
            t = step / frames_per_transition
            z = lerp(z_start, z_end, t)
            flat_img = decode(z, decoder_weights, decoder_biases)
            gray = flat_img.reshape(IMG_SIZE, IMG_SIZE).astype(np.float32)
            gray = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC)
            gray = np.clip(gray, 0, 1)
            if colorful:
                frame = colorize(gray, len(frames) / total)      # RGB
            else:
                g8 = (gray * 255).astype(np.uint8)
                frame = np.stack([g8, g8, g8], axis=-1)
            frames.append(frame)
    n_frames_written = len(frames)

    # 2) encode as H.264 (plays in every browser). imageio-ffmpeg bundles
    #    its own ffmpeg, so nothing extra needs installing.
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            video_path, fps=FPS, codec="libx264",
            pixelformat="yuv420p", macro_block_size=None,
        )
        for f in frames:
            writer.append_data(f)
        writer.close()
    except Exception:
        # Fallback: OpenCV mp4v (plays in VLC / Windows player, not browsers)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_SIZE, FRAME_SIZE), isColor=True)
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()

    return {
        "video_path": video_path,
        "n_frames": n_frames_written,
        "fps": FPS,
        "duration_sec": round(n_frames_written / FPS, 2),
        "n_keyframes": len(keyframes) - 1,
    }


if __name__ == "__main__":
    info = render_video()
    print(f"Video saved to: {info['video_path']}")
    print(f"Frames: {info['n_frames']} | FPS: {info['fps']} | "
          f"Duration: {info['duration_sec']}s | Keyframe transitions: {info['n_keyframes']}")
