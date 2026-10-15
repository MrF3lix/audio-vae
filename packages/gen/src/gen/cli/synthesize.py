"""Text -> wav with a trained generative model, its VAE and the vocoder.

    uv run gen-synthesize --ckpt logs/train_gen/fm_be/runs/<date>/checkpoints/best.pt \
        --text "Guten Morgen, wie geht es Ihnen?" --out synth/
    uv run gen-synthesize --ckpt ... --filelist data/filelists/be/val.txt --limit 10 --out synth/

Writes `<out>/<index>.wav` (or `<stem>.wav` for filelist items) and `<out>/texts.txt`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import OmegaConf

from common.data.filelist import parse_filelist
from common.training.utils import load_checkpoint, resolve_device
from vae import load_pretrained


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--text", action="append", default=[], help="text to synthesise (repeatable)")
    p.add_argument("--filelist", type=Path, help="`path|text` filelist; output files are named by stem")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vae-ckpt", type=Path, default=None, help="default: the one the model was trained with")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--method", default=None, choices=["euler", "midpoint"])
    p.add_argument("--length-scale", type=float, default=1.0, help=">1 speaks slower")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.ckpt)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = hydra.utils.instantiate(OmegaConf.to_container(cfg.model, resolve=True))
    model.load_state_dict(ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"])
    model = model.to(device).eval()
    vae = load_pretrained(args.vae_ckpt or cfg.vae_ckpt, device)
    vocoder_cfg = OmegaConf.to_container(cfg.vocoder, resolve=True)
    vocoder_cfg["device"] = str(device)
    vocoder = hydra.utils.instantiate(vocoder_cfg)

    items = [(f"{i:03d}", t) for i, t in enumerate(args.text)]
    if args.filelist:
        items += [(u.path.stem, u.text) for u in parse_filelist(args.filelist)]
    items = items[: args.limit]
    if not items:
        p.error("nothing to synthesise: pass --text and/or --filelist")

    sampling = cfg.get("sampling", {})
    kwargs = dict(
        n_steps=args.n_steps or sampling.get("n_steps", 32),
        guidance_scale=args.guidance_scale if args.guidance_scale is not None else sampling.get("guidance_scale", 2.0),
        method=args.method or sampling.get("method", "midpoint"),
        length_scale=args.length_scale,
        temperature=args.temperature,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    with open(args.out / "texts.txt", "w", encoding="utf-8") as f:
        for start in range(0, len(items), args.batch_size):
            chunk = items[start : start + args.batch_size]
            out = model.synthesise([t for _, t in chunk], generator=gen, **kwargs)
            for i, (name, text) in enumerate(chunk):
                L = int(out["z_lengths"][i])
                mel = vae.decode(out["z"][i : i + 1, :, :L], out["z_lengths"][i : i + 1])
                wav = vocoder(vae.denormalize_mel(mel))[0].cpu().numpy()
                sf.write(args.out / f"{name}.wav", wav, vocoder.sample_rate)
                f.write(f"{name}|{text}\n")
                print(f"{name}.wav  {len(wav) / vocoder.sample_rate:.2f}s  {text}")


if __name__ == "__main__":
    main()
