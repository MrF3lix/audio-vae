"""Text encoders: raw strings -> features `(B, N, D)` and a padding mask `(B, N)`.

Both own their tokenisation, so a model can be called with plain text at inference and its
checkpoint is self-contained.

* `CharTextEncoder`: characters of the cleaned text, trained from scratch (no downloads).
* `PretrainedTextEncoder`: a frozen Hugging Face encoder (e.g. ByT5, German BERT) plus a small
  trainable adapter. The frozen weights are kept out of the module tree (a process-wide cache),
  so they are neither saved in checkpoints nor copied by EMA.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from common.lengths import sequence_mask
from gen.model.modules import ConvPrenet, RotaryEmbedding, TransformerBlock
from gen.text.cleaners import get_cleaner

# Every character of the cleaned (`swiss_german`) Bernese corpus.
DEFAULT_SYMBOLS = " !\"',-.:;?abcdefghijklmnopqrstuvwxyzàâäçèéêöü"


class CharTextEncoder(nn.Module):
    PAD, UNK = 0, 1

    def __init__(self, dim: int = 256, n_layers: int = 4, n_heads: int = 4, ff_mult: int = 4, kernel_size: int = 3,
                 prenet_layers: int = 3, dropout: float = 0.1, cleaner: str = "swiss_german", symbols: str | None = None):
        super().__init__()
        self.dim = dim
        self.cleaner = get_cleaner(cleaner)
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.index = {c: i + 2 for i, c in enumerate(self.symbols)}
        self.embed = nn.Embedding(len(self.symbols) + 2, dim, padding_idx=self.PAD)
        nn.init.normal_(self.embed.weight, 0.0, dim**-0.5)
        self.prenet = ConvPrenet(dim, prenet_layers) if prenet_layers else None
        self.rope = RotaryEmbedding(dim // n_heads)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, ff_mult, kernel_size, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def tokenize(self, texts: list[str], device=None) -> tuple[Tensor, Tensor]:
        seqs = [[self.index.get(c, self.UNK) for c in self.cleaner(t)] or [self.UNK] for t in texts]
        lengths = torch.tensor([len(s) for s in seqs])
        ids = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        return ids.to(device), lengths.to(device)

    def unknown_chars(self, text: str) -> set[str]:
        return {c for c in self.cleaner(text) if c not in self.index}

    def forward(self, ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        mask = sequence_mask(lengths, ids.shape[1])
        x = self.embed(ids) * mask[..., None]
        if self.prenet is not None:
            x = self.prenet(x, mask)
        rope = self.rope(x.shape[1], x.device)
        for block in self.blocks:
            x = block(x, mask, rope)
        return self.norm(x) * mask[..., None], mask

    def encode(self, texts: list[str], device=None) -> tuple[Tensor, Tensor]:
        return self(*self.tokenize(texts, device))


_FROZEN: dict[tuple[str, str], tuple[object, nn.Module]] = {}


def _load_frozen(name: str, device) -> tuple[object, nn.Module]:
    key = (name, str(device))
    if key not in _FROZEN:
        from transformers import AutoModel, AutoTokenizer  # pylint: disable=import-outside-toplevel

        def load(cls):
            try:  # prefer the local Hub cache (no HEAD requests on a flaky connection)
                return cls.from_pretrained(name, local_files_only=True)
            except OSError:
                return cls.from_pretrained(name)

        tokenizer = load(AutoTokenizer)
        model = load(AutoModel)
        if getattr(model.config, "is_encoder_decoder", False):
            model = model.get_encoder()
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _FROZEN[key] = (tokenizer, model)
    return _FROZEN[key]


class PretrainedTextEncoder(nn.Module):
    def __init__(self, name: str = "google/byt5-small", dim: int = 256, adapter_layers: int = 2, n_heads: int = 4,
                 ff_mult: int = 4, dropout: float = 0.1, cleaner: str = "swiss_german_cased", hidden_size: int | None = None):
        super().__init__()
        self.name, self.dim = name, dim
        self.cleaner = get_cleaner(cleaner)
        if hidden_size is None:
            from transformers import AutoConfig  # pylint: disable=import-outside-toplevel

            cfg = AutoConfig.from_pretrained(name)
            hidden_size = getattr(cfg, "d_model", None) or cfg.hidden_size
        self.proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))
        self.rope = RotaryEmbedding(dim // n_heads)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, ff_mult, 1, dropout) for _ in range(adapter_layers)])
        self.norm = nn.LayerNorm(dim)

    def tokenize(self, texts: list[str], device=None) -> tuple[Tensor, Tensor]:
        tokenizer, _ = _load_frozen(self.name, device or "cpu")
        enc = tokenizer([self.cleaner(t) for t in texts], padding=True, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].sum(1).to(device)

    def forward(self, ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        _, model = _load_frozen(self.name, ids.device)
        mask = sequence_mask(lengths, ids.shape[1])
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        x = self.proj(h.float()) * mask[..., None]
        rope = self.rope(x.shape[1], x.device)
        for block in self.blocks:
            x = block(x, mask, rope)
        return self.norm(x) * mask[..., None], mask

    def encode(self, texts: list[str], device=None) -> tuple[Tensor, Tensor]:
        return self(*self.tokenize(texts, device))
