## Audio VAE + latent generative models

A uv workspace with three packages:

| package | contents | depends on |
|---|---|---|
| [`common`](packages/common) | mel front-end, vocoder wrapper, filelists, mel dataset, training loop, loggers, shared hydra config groups (`trainer`, `optimizer`, `logger`, `paths`, `hydra`, `vocoder`) | – |
| [`vae`](packages/vae) | mel VAE, latent statistics, latent cache, `load_pretrained` | `common` |
| [`gen`](packages/gen) | text-conditioned generative models in the VAE latent space (flow matching) | `common`, `vae` |

`gen` only uses the VAE through `vae.load_pretrained(ckpt)` and the latent cache (`vae.latents`),
so the VAE stays usable on its own.

```bash
uv sync                       # all packages, editable
```

### 1. VAE

```bash
uv run vae-train experiment=vae_be
uv run vae-train experiment=vae_be ckpt_path=logs/train_vae/vae_be/runs/<date>/checkpoints/last.pt   # continue
```

Then fit the per-channel latent statistics (required by everything downstream; `train.submit`
does this automatically):

```bash
uv run vae-latent-stats --ckpt logs/train_vae/vae_be/runs/<date>/checkpoints/best.pt
```

### 2. Latent cache (optional; `gen-train` builds it on first use)

```bash
uv run vae-encode-latents --ckpt logs/train_vae/vae_be/runs/<date>/checkpoints/best.pt \
    --filelists data/filelists/be/train.txt data/filelists/be/val.txt --check 20
```

Writes `data/cache/latents/<vae run>_<ckpt>_step<k>/`: one `(2, C, L)` `.npy` per utterance
(normalised posterior mean and log-variance) and `meta.json`. `--check` decodes cached latents
and reports the reconstruction L1, which should match the VAE's `val/recon`.

### 3. Text-conditioned flow matching

```bash
uv run gen-train experiment=fm_be                                  # char encoder, trained from scratch
uv run gen-train experiment=fm_be model/text_encoder=byt5          # frozen ByT5-small + adapter
uv run gen-train experiment=fm_be vae_ckpt=<path/to/vae/best.pt>   # another VAE
```

Cluster: `sbatch train_gen.submit` (`VAE_CKPT=<path>` to override the VAE).

Model: text encoder → cross-attention into a transformer over latent frames (time via adaLN-zero),
plus a total-length predictor. No monotonic alignment, since the Standard German text and the
Bernese speech differ in word order and vocabulary. Validation logs `val/fm` (flow loss at fixed
noise), `val/dur_abs_err_s` (length error in seconds), mels and audio of generated samples.

Synthesis:

```bash
uv run gen-synthesize --ckpt logs/train_gen/fm_be/runs/<date>/checkpoints/best.pt \
    --text "Guten Morgen, wie geht es Ihnen?" --out synth/
uv run gen-synthesize --ckpt ... --filelist data/filelists/be/val.txt --limit 10 --guidance-scale 3 --out synth/
```
