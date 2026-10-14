## Audio VAE


Training the VAE

```bash
uv run vae-train experiment=vae_be
```

Continue training


```bash
uv run vae-train experiment=vae_be ckpt_path=logs/train_vae/vae_be/runs/2026-10-14_11-32-37/checkpoints/last.pt

```