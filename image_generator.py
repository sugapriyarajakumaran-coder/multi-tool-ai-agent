"""
image_generator.py
-------------------
A small deep-learning image generator built from scratch with an
autoencoder architecture:

    input image (32x32=1024) -> encoder -> latent (8-d) -> decoder -> image (1024)

Trained with scikit-learn's MLPRegressor (a real feed-forward neural
network trained by backprop / Adam) in an autoencoder configuration
(hidden_layer_sizes with a bottleneck). After training we split the
network's own learned weight matrices at the bottleneck layer so we can
run the DECODER half on brand-new, never-seen random latent vectors --
this is how we "generate" new images instead of just reconstructing
training data (the same core idea behind GAN/VAE decoders, simplified).

No internet / pretrained weights are used -- the training data is
generated procedurally (circles, squares, triangles at random positions,
sizes and rotations) so the whole pipeline is self-contained.
"""

import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from PIL import Image

IMG_SIZE = 32
LATENT_DIM = 8
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "images"))


# ---------------------------------------------------------------------
# 1. Synthetic training data generator
# ---------------------------------------------------------------------
def make_shape_image(rng, kind=None):
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cx, cy = rng.integers(10, 22, size=2)
    size = rng.integers(6, 12)
    kind = kind or rng.choice(["circle", "square", "triangle"])

    if kind == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= size ** 2
    elif kind == "square":
        mask = (np.abs(xx - cx) <= size) & (np.abs(yy - cy) <= size)
    else:  # triangle
        mask = (yy - cy >= -size) & (yy <= cy + size) & \
               (np.abs(xx - cx) <= (cy + size - yy) * 0.8)

    img[mask] = 1.0
    return img


def build_dataset(n_samples=400, seed=0):
    rng = np.random.default_rng(seed)
    data = np.stack([make_shape_image(rng) for _ in range(n_samples)])
    return data.reshape(n_samples, -1)  # flatten to (N, 1024)


# ---------------------------------------------------------------------
# 2. Train the autoencoder (this IS the "deep learning" step)
# ---------------------------------------------------------------------
def train_autoencoder(X, seed=0):
    model = MLPRegressor(
        hidden_layer_sizes=(128, LATENT_DIM, 128),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=seed,
        early_stopping=False,
        alpha=1e-4,
    )
    model.fit(X, X)  # autoencoder: target = input
    return model


def split_encoder_decoder(model: MLPRegressor):
    """
    MLPRegressor doesn't expose a decode-only API, so we copy its learned
    weight matrices and manually run the second half of the forward pass
    (bottleneck -> output). This lets us feed in ARBITRARY latent vectors
    (not just ones produced by the encoder) to generate new images.
    """
    # Layer topology for hidden_layer_sizes=(128, LATENT_DIM, 128):
    #   input(1024) --W0--> h1(128) --W1--> bottleneck(8) --W2--> h2(128) --W3--> output(1024)
    # coefs_ = [W0, W1, W2, W3]. Encoder = W0,W1 (input->bottleneck).
    # Decoder = W2,W3 (bottleneck->output) -- this is the half we reuse for generation.
    weights = model.coefs_          # list of weight matrices per layer
    biases = model.intercepts_      # list of bias vectors per layer
    decoder_weights = weights[2:]
    decoder_biases = biases[2:]
    return decoder_weights, decoder_biases


def relu(x):
    return np.maximum(0, x)


def decode(latent_vec, decoder_weights, decoder_biases):
    a = latent_vec
    n_layers = len(decoder_weights)
    for i, (W, b) in enumerate(zip(decoder_weights, decoder_biases)):
        a = a @ W + b
        if i < n_layers - 1:
            a = relu(a)
    return np.clip(a, 0, 1)


# ---------------------------------------------------------------------
# 3. Generation: sample new latent vectors and decode them into images
# ---------------------------------------------------------------------
def generate_images(n_images=6, seed=42):
    os.makedirs(OUT_DIR, exist_ok=True)

    X = build_dataset(n_samples=400, seed=0)
    model = train_autoencoder(X, seed=0)

    # learn the real distribution of latents produced by the encoder so
    # our random samples land in a "valid" region of latent space
    # get true latent activations via a manual forward pass through encoder half
    encoder_weights = model.coefs_[:2]
    encoder_biases = model.intercepts_[:2]
    a = X
    for W, b in zip(encoder_weights, encoder_biases):
        a = relu(a @ W + b)
    latents = a  # shape (N, LATENT_DIM)
    lat_mean, lat_std = latents.mean(axis=0), latents.std(axis=0) + 1e-3

    decoder_weights, decoder_biases = split_encoder_decoder(model)

    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_images):
        z = lat_mean + rng.normal(size=LATENT_DIM) * lat_std * 1.2
        z = relu(z)  # keep consistent with relu-activated latent space
        flat_img = decode(z, decoder_weights, decoder_biases)
        img_arr = (flat_img.reshape(IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L").resize((128, 128), Image.NEAREST)
        path = os.path.join(OUT_DIR, f"generated_{i:02d}.png")
        img.save(path)
        paths.append(path)

    # also save a contact sheet of a few training examples for comparison
    sample_train = X[:6].reshape(-1, IMG_SIZE, IMG_SIZE)
    sheet = Image.new("L", (128 * 6, 128))
    for i, arr in enumerate(sample_train):
        tile = Image.fromarray((arr * 255).astype(np.uint8), mode="L").resize((128, 128), Image.NEAREST)
        sheet.paste(tile, (i * 128, 0))
    sheet_path = os.path.join(OUT_DIR, "training_samples.png")
    sheet.save(sheet_path)

    return {
        "training_loss": float(model.loss_),
        "n_training_images": len(X),
        "latent_dim": LATENT_DIM,
        "generated_paths": paths,
        "training_sample_path": sheet_path,
        "model": model,
        "decoder_weights": decoder_weights,
        "decoder_biases": decoder_biases,
        "latent_stats": (lat_mean, lat_std),
    }


if __name__ == "__main__":
    result = generate_images(n_images=6)
    print(f"Final training loss (MSE): {result['training_loss']:.5f}")
    print(f"Trained on {result['n_training_images']} synthetic shape images")
    print(f"Latent space dimension: {result['latent_dim']}")
    print("Generated images:")
    for p in result["generated_paths"]:
        print(" -", p)
    print("Training sample sheet:", result["training_sample_path"])
