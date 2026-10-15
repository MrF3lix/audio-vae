"""Train a text-conditioned generative model on the VAE's latents.

    uv run gen-train experiment=fm_be     # flow matching
    uv run gen-train experiment=em_be     # energy matching
"""

import logging
from pathlib import Path

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from common.lengths import sequence_mask
from common.training.audio import AudioLogger
from common.training.loggers import build_logger
from common.training.trainer import Trainer
from common.training.utils import count_parameters, plot_mel, resolve_device, seed_everything

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")
log = logging.getLogger(__name__)


def make_validate(loader, vae, device, logger, audio, cfg):
    """Validation: the model's losses at fixed noise and times (comparable across runs), length
    error, and `n_audio_samples` syntheses from the first validation texts, decoded and vocoded."""
    n_samples, audio_every = cfg.trainer.n_audio_samples, cfg.trainer.audio_every_n_vals
    sampling = OmegaConf.to_container(cfg.sampling, resolve=True)
    seconds_per_frame = vae.compression * vae.mel.hop_length / vae.mel.sample_rate

    @torch.no_grad()
    def validate(model, step, n_val):
        gen = torch.Generator().manual_seed(0)
        sums, dur_sum, err_sum, n_frames, n_utts = {}, 0.0, 0.0, 0, 0
        refs = []
        for batch in loader:
            z, z_lengths = batch["z"].to(device), batch["z_lengths"].to(device)
            mask = sequence_mask(z_lengths, z.shape[-1])
            h, h_mask = model.encode_text(batch["texts"], device)
            noise = torch.randn(z.shape, generator=gen).to(device)
            t = torch.rand(z.shape[0], generator=gen).to(device)
            frames = int(z_lengths.sum())
            for key, value in model.validation_losses(z, mask, h, h_mask, noise, t).items():
                sums[key] = sums.get(key, 0.0) + float(value) * frames
            n_frames += frames
            log_pred = model.duration(h, h_mask)
            dur_sum += float(model.duration.loss(log_pred, z_lengths)) * z.shape[0]
            err_sum += float((log_pred.exp() - z_lengths).abs().sum())
            n_utts += z.shape[0]
            for i in range(z.shape[0]):
                if len(refs) < n_samples:
                    refs.append((batch["texts"][i], z[i, :, : z_lengths[i]]))

        if refs:
            texts = [text for text, _ in refs]
            out = model.synthesise(texts, generator=torch.Generator().manual_seed(1), **sampling)
            generated, references = [], []
            for i, (_, z_ref) in enumerate(refs):
                L = int(out["z_lengths"][i])
                mel_gen = vae.decode(out["z"][i : i + 1, :, :L], out["z_lengths"][i : i + 1])[0].cpu()
                mel_ref = vae.decode(z_ref[None], torch.tensor([z_ref.shape[-1]], device=device))[0].cpu()
                generated.append(mel_gen)
                references.append(mel_ref)
                logger.log_image(f"mel/{i}", plot_mel(mel_ref.numpy(), mel_gen.numpy(),
                                                     titles=["reference (VAE)", f"generated: {texts[i][:80]}"]), step)
            if audio_every > 0 and n_val % audio_every == 0:
                audio(logger, step, {f"audio/reference_{i}": m for i, m in enumerate(references)}, once=True)
                audio(logger, step, {f"audio/generated_{i}": m for i, m in enumerate(generated)})

        metrics = {f"val/{k}": v / max(n_frames, 1) for k, v in sums.items()}
        metrics.update({
            "val/dur": dur_sum / max(n_utts, 1),
            "val/dur_abs_err_s": err_sum / max(n_utts, 1) * seconds_per_frame,
        })
        return metrics

    return validate


def train(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = resolve_device(cfg.trainer.device)
    log.info("Device: %s", device)

    datamodule = hydra.utils.instantiate(cfg.data)
    with open_dict(cfg):
        cfg.model.latent_dim = datamodule.latent_dim  # from the VAE; stored in the checkpoint config
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    model = hydra.utils.instantiate(cfg.model).to(device)
    log.info("Parameters: %.2f M (trainable %.2f M)", count_parameters(model) / 1e6,
             sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    optimizer = hydra.utils.instantiate(cfg.optimizer)(p for p in model.parameters() if p.requires_grad)
    logger = build_logger(cfg)

    tcfg = cfg.trainer
    vae = datamodule.vae
    audio = AudioLogger(cfg.vocoder, vae.denormalize_mel, tcfg.audio_every_n_vals > 0, device)
    validate = make_validate(datamodule.valid_dataloader(), vae, device, logger, audio, cfg)

    trainer = Trainer(cfg, model, optimizer, datamodule.train_dataloader(), device, logger, validate,
                      extra_state={"latent_dir": str(datamodule.latent_dir)})
    if cfg.ckpt_path:
        trainer.resume(cfg.ckpt_path)

    best = trainer.fit()
    logger.close()
    return best


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train.yaml")
def main(cfg: DictConfig):
    return train(cfg)


if __name__ == "__main__":
    main()
