"""Standalone AutoPCP audio-pair scorer.

This module follows the public Stopes AutoPCP comparator behaviour without
depending on Stopes, Hydra, Submitit, or the Stopes audio loading stack.
"""

from __future__ import annotations

import logging
import math
import os
import typing as tp
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


LOGGER = logging.getLogger(__name__)

DEFAULT_COMPARATOR_URL = (
    "https://dl.fbaipublicfiles.com/speech_expressivity_evaluation/"
    "AutoPCP-multilingual-v2.zip"
)
DEFAULT_COMPARATOR_NAME = "AutoPCP-multilingual-v2"
DEFAULT_ENCODER = "facebook/wav2vec2-large-xlsr-53"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_PICK_LAYER = 9
INFTY = 1e6


class Comparator(nn.Module):
    """Comparator architecture used by the released AutoPCP checkpoint."""

    def __init__(
        self,
        idim: int,
        odim: int,
        nhid: list[int],
        dropout: float,
        activation: str,
        input_form: str,
        norm_emb: bool,
        output_act: bool,
        trainable_pooler: bool = False,
        use_gpu: bool | None = None,
    ) -> None:
        super().__init__()
        self.input_form = input_form
        self.dropout = dropout
        self.idim = idim
        self.odim = odim
        self.norm_emb = norm_emb
        self.nhid = nhid
        self.activation = activation
        self.output_act = output_act
        self.trainable_pooler = trainable_pooler
        self.use_gpu = bool(use_gpu)

        self.pooler: nn.Module | None = None
        if trainable_pooler:
            self.pooler = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(idim, 1))

        if input_form == "comet":
            mlp_idim = 6 * idim
        elif input_form == "qe":
            mlp_idim = 4 * idim
        else:
            raise ValueError(f"unrecognized comparator input_form: {input_form}")

        modules: list[nn.Module] = []
        if nhid:
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            previous = mlp_idim
            for hidden_size in nhid:
                if hidden_size <= 0:
                    continue
                modules.append(nn.Linear(previous, hidden_size))
                previous = hidden_size
                if activation == "TANH":
                    modules.append(nn.Tanh())
                elif activation == "RELU":
                    modules.append(nn.ReLU())
                else:
                    raise ValueError(f"unrecognized comparator activation: {activation}")
                if dropout > 0:
                    modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(previous, odim))
            if output_act:
                modules.append(nn.Tanh())
        else:
            modules.append(nn.Linear(mlp_idim, odim))
        self.mlp = nn.Sequential(*modules)

    @classmethod
    def load(cls, config_path: str | Path, device: torch.device) -> "Comparator":
        config_path = Path(config_path).expanduser().resolve()
        if config_path.is_dir():
            configs = sorted(config_path.rglob("*.config"))
            if not configs:
                raise FileNotFoundError(f"no .config file found under {config_path}")
            config_path = configs[0]

        config = _torch_load(config_path, map_location="cpu")
        if not isinstance(config, dict):
            raise TypeError(f"unexpected comparator config type: {type(config)}")
        config.pop("use_gpu", None)
        model = cls(**config)
        weights_path = config_path.with_suffix(".pt")
        if not weights_path.exists():
            raise FileNotFoundError(f"comparator weights not found: {weights_path}")
        state = _torch_load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model

    def forward(
        self,
        src: torch.Tensor,
        ref: torch.Tensor | None = None,
        mt: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        mt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src = self._pool(src, src_mask)
        ref = self._pool(ref, ref_mask)
        mt = self._pool(mt, mt_mask)
        processed = self._process_input(
            self._norm_vec(src),
            self._norm_vec(ref),
            self._norm_vec(mt),
        )
        return self.mlp(processed)

    def _norm_vec(self, emb: torch.Tensor | None) -> torch.Tensor | None:
        if self.norm_emb and emb is not None:
            return F.normalize(emb)
        return emb

    def _pool(
        self, emb: torch.Tensor | None, mask: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if emb is None or self.pooler is None:
            return emb
        if emb.ndim != 3:
            return emb
        logits = self.pooler(emb).squeeze(-1)
        if mask is None:
            mask = ((emb != 0).any(dim=-1)).to(torch.long)
        logits = logits - (1 - mask.to(logits.device)) * INFTY
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("btd,bt->bd", emb, weights)

    def _process_input(
        self,
        src: torch.Tensor,
        ref: torch.Tensor | None = None,
        mt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_form == "comet":
            if ref is None or mt is None:
                raise ValueError("comet comparator requires src, ref, and mt")
            return torch.cat(
                [
                    ref,
                    mt,
                    src * mt,
                    ref * mt,
                    torch.abs(mt - src),
                    torch.abs(mt - ref),
                ],
                dim=-1,
            )
        if self.input_form == "qe":
            if mt is None:
                mt = ref
            if mt is None:
                raise ValueError("qe comparator requires src and mt/ref")
            return torch.cat([src, mt, src * mt, torch.abs(mt - src)], dim=-1)
        raise ValueError(f"unrecognized comparator input_form: {self.input_form}")


class AutoPCP:
    """AutoPCP pair scorer.

    Example:
        scorer = AutoPCP(device="cuda")
        scores = scorer.score_pairs(source_paths, target_paths)
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        batch_size: int = 16,
        cache_dir: str | Path | None = None,
        comparator_path: str | Path | None = None,
        comparator_url: str = DEFAULT_COMPARATOR_URL,
        encoder_path: str = DEFAULT_ENCODER,
        pick_layer: int = DEFAULT_PICK_LAYER,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        symmetrize: bool = True,
        progress: bool = False,
    ) -> None:
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.encoder_path = encoder_path
        self.pick_layer = pick_layer
        self.sample_rate = sample_rate
        self.symmetrize = symmetrize
        self.progress = progress

        if comparator_path is None:
            comparator_path = ensure_comparator_checkpoint(
                cache_dir=cache_dir,
                url=comparator_url,
            )
        self.comparator_path = Path(comparator_path).expanduser().resolve()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(encoder_path)
        self.encoder = Wav2Vec2Model.from_pretrained(encoder_path).to(self.device)
        self.encoder.eval()
        self.comparator = Comparator.load(self.comparator_path, device=self.device)

    @torch.inference_mode()
    def score_pairs(
        self,
        source_paths: tp.Sequence[str | Path],
        target_paths: tp.Sequence[str | Path],
    ) -> list[float]:
        if len(source_paths) != len(target_paths):
            raise ValueError(
                f"source/target length mismatch: {len(source_paths)} != {len(target_paths)}"
            )
        if not source_paths:
            return []

        src_emb = self.encode_paths(source_paths)
        tgt_emb = self.encode_paths(target_paths)
        scores = self._predict(src_emb, tgt_emb)
        if self.symmetrize:
            reverse = self._predict(tgt_emb, src_emb)
            scores = (scores + reverse) / 2
        return [float(score) for score in scores.cpu().tolist()]

    @torch.inference_mode()
    def encode_paths(self, audio_paths: tp.Sequence[str | Path]) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        iterator = range(0, len(audio_paths), self.batch_size)
        if self.progress:
            iterator = tqdm(iterator, desc="AutoPCP encode", unit="batch")

        original_layers: list[nn.Module] | None = None
        if isinstance(self.encoder, Wav2Vec2Model) and self.pick_layer is not None:
            original_layers = list(self.encoder.encoder.layers)
            self.encoder.encoder.layers = nn.ModuleList(
                original_layers[: self.pick_layer + 1]
            )
        try:
            for start in iterator:
                paths = audio_paths[start : start + self.batch_size]
                audios = [
                    load_audio(path, target_sample_rate=self.sample_rate)
                    for path in paths
                ]
                inputs = self.feature_extractor(
                    audios,
                    sampling_rate=self.sample_rate,
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output = self.encoder(**inputs, output_hidden_states=True)
                states = output.hidden_states[self.pick_layer]
                frame_mask = _feature_attention_mask(
                    self.encoder,
                    states.shape[1],
                    inputs["attention_mask"],
                ).to(states.device)
                pooled = masked_temporal_mean(states, frame_mask)
                batches.append(pooled.cpu())
        finally:
            if original_layers is not None:
                self.encoder.encoder.layers = nn.ModuleList(original_layers)
        return torch.cat(batches, dim=0)

    @torch.inference_mode()
    def _predict(self, src_emb: torch.Tensor, tgt_emb: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, src_emb.shape[0], self.batch_size):
            src = src_emb[start : start + self.batch_size].to(self.device)
            tgt = tgt_emb[start : start + self.batch_size].to(self.device)
            outputs.append(self.comparator(src=src, mt=tgt)[:, 0].detach().cpu())
        return torch.cat(outputs, dim=0)


def ensure_comparator_checkpoint(
    cache_dir: str | Path | None = None,
    url: str = DEFAULT_COMPARATOR_URL,
) -> Path:
    """Download and extract the comparator checkpoint if needed."""
    root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    target_dir = root / DEFAULT_COMPARATOR_NAME
    existing = _find_config(target_dir)
    if existing is not None:
        return existing

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{DEFAULT_COMPARATOR_NAME}.zip"
    if not zip_path.exists():
        partial = zip_path.with_suffix(".zip.part")
        LOGGER.info("downloading AutoPCP comparator checkpoint from %s", url)
        urllib.request.urlretrieve(url, partial)
        partial.replace(zip_path)

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        _safe_extract(archive, target_dir)

    config = _find_config(target_dir)
    if config is None:
        raise FileNotFoundError(f"no .config found after extracting {zip_path}")
    return config


def default_cache_dir() -> Path:
    if os.environ.get("AUTO_PCP_CACHE_DIR"):
        return Path(os.environ["AUTO_PCP_CACHE_DIR"]).expanduser()
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "autopcp"
    return Path.home() / ".cache" / "autopcp"


def load_audio(path: str | Path, target_sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Load an audio file, convert to mono, resample, and return float32 samples."""
    path = Path(path).expanduser()
    try:
        waveform, sample_rate = _load_with_torchaudio(path)
    except Exception:
        waveform, sample_rate = _load_with_soundfile(path)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)
    elif waveform.ndim != 1:
        raise ValueError(f"unsupported audio shape for {path}: {waveform.shape}")
    if waveform.size == 0:
        raise ValueError(f"empty audio file: {path}")

    waveform = waveform.astype(np.float32, copy=False)
    if sample_rate != target_sample_rate:
        waveform = _resample(waveform, sample_rate, target_sample_rate)
    return waveform.astype(np.float32, copy=False)


def masked_temporal_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool frame states without including padded frames."""
    weights = mask.to(dtype=states.dtype).unsqueeze(-1)
    numerator = (states * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _torch_load(path: Path, map_location: str | torch.device | None = None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return resolved


def _load_with_torchaudio(path: Path) -> tuple[np.ndarray, int]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.detach().cpu().numpy(), int(sample_rate)


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    array = np.asarray(data)
    if array.ndim == 2:
        array = array.T
    return array, int(sample_rate)


def _resample(
    waveform: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    try:
        import torchaudio.functional as F_audio

        tensor = torch.from_numpy(waveform).unsqueeze(0)
        resampled = F_audio.resample(tensor, source_sample_rate, target_sample_rate)
        return resampled.squeeze(0).numpy()
    except Exception:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_sample_rate, target_sample_rate)
        up = target_sample_rate // divisor
        down = source_sample_rate // divisor
        return resample_poly(waveform, up, down).astype(np.float32, copy=False)


def _feature_attention_mask(
    model: Wav2Vec2Model,
    feature_len: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    try:
        return model._get_feature_vector_attention_mask(
            feature_len,
            attention_mask,
            add_adapter=False,
        )
    except TypeError:
        return model._get_feature_vector_attention_mask(feature_len, attention_mask)


def _find_config(root: Path) -> Path | None:
    if not root.exists():
        return None
    configs = sorted(root.rglob("*.config"))
    return configs[0] if configs else None


def _safe_extract(archive: zipfile.ZipFile, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    for member in archive.infolist():
        destination = (target_dir / member.filename).resolve()
        if destination != target_dir and target_dir not in destination.parents:
            raise ValueError(f"unsafe zip member path: {member.filename}")
    archive.extractall(target_dir)


__all__ = [
    "AutoPCP",
    "Comparator",
    "DEFAULT_COMPARATOR_URL",
    "DEFAULT_ENCODER",
    "DEFAULT_PICK_LAYER",
    "DEFAULT_SAMPLE_RATE",
    "ensure_comparator_checkpoint",
    "load_audio",
    "masked_temporal_mean",
]
