from __future__ import annotations

from importlib import import_module

from .interfaces import SpeakerEmbeddingModel

# Stable IDs used across CLI/config/sidecar metadata.
EMBEDDING_MODEL_IDS = ("resemblyzer", "tfgridnet", "beamformer_resemblyzer")

_MODEL_MODULES: dict[str, str] = {
    "resemblyzer": ".ResemblyzerSpeakerEmbeddingModel",
    "tfgridnet": ".TFGridNetSpeakerEmbeddingModel",
    "beamformer_resemblyzer": ".BeamformerResemblyzerSpeakerEmbeddingModel",
}

_MODEL_CLASS_NAMES: dict[str, str] = {
    "resemblyzer": "ResemblyzerSpeakerEmbeddingModel",
    "tfgridnet": "TFGridNetSpeakerEmbeddingModel",
    "beamformer_resemblyzer": "BeamformerResemblyzerSpeakerEmbeddingModel",
}


def _normalized_id(model_id: str) -> str:
    normalized = model_id.strip().lower()
    if normalized not in EMBEDDING_MODEL_IDS:
        allowed = ", ".join(EMBEDDING_MODEL_IDS)
        raise ValueError(f"Unknown embedding model '{model_id}'. Allowed: {allowed}")
    return normalized


def _load_model_class(model_id: str) -> type[SpeakerEmbeddingModel]:
    normalized = _normalized_id(model_id)
    module = import_module(_MODEL_MODULES[normalized], package=__package__)
    class_name = _MODEL_CLASS_NAMES[normalized]
    return getattr(module, class_name)


def create_embedding_model(model_id: str) -> SpeakerEmbeddingModel:
    """Instantiate a speaker embedding model by stable model ID."""
    model_cls = _load_model_class(model_id)
    return model_cls()


def embedding_model_class_name(model_id: str) -> str:
    """Return the concrete class name for a stable model ID."""
    return _MODEL_CLASS_NAMES[_normalized_id(model_id)]
