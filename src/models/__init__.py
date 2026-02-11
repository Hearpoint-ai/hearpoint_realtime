from datetime import datetime
from enum import Enum
from pathlib import Path


class RecordingKind(str, Enum):
    MIXTURE = "mixture"
    EXTRACTED = "extracted"


class Speaker:
    def __init__(
        self,
        id: str,
        name: str,
        created_at: datetime,
        embedding_path: Path,
        enrollment_audio_path: Path,
        processing_ms: float | None = None,
    ):
        self.id = id
        self.name = name
        self.created_at = created_at
        self.embedding_path = embedding_path
        self.enrollment_audio_path = enrollment_audio_path
        self.processing_ms = processing_ms

    def to_dict(self, media_root: Path) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "embedding_path": str(self.embedding_path.relative_to(media_root)),
            "enrollment_audio_path": str(self.enrollment_audio_path.relative_to(media_root)),
            "processing_ms": self.processing_ms,
        }

    @classmethod
    def from_dict(cls, data: dict, media_root: Path) -> "Speaker":
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            embedding_path=media_root / data["embedding_path"],
            enrollment_audio_path=media_root / data["enrollment_audio_path"],
            processing_ms=data.get("processing_ms"),
        )


class Recording:
    def __init__(
        self,
        id: str,
        name: str,
        kind: RecordingKind,
        created_at: datetime,
        file_path: Path,
        cosine_similarity_before: float | None = None,
        cosine_similarity_after: float | None = None,
        cosine_similarity_series_after: list[float] | None = None,
    ):
        self.id = id
        self.name = name
        self.kind = kind
        self.created_at = created_at
        self.file_path = file_path
        self.cosine_similarity_before = cosine_similarity_before
        self.cosine_similarity_after = cosine_similarity_after
        self.cosine_similarity_series_after = cosine_similarity_series_after

    def to_dict(self, media_root: Path) -> dict:
        payload = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind.value,
            "created_at": self.created_at.isoformat(),
            "file_path": str(self.file_path.relative_to(media_root)),
        }
        if self.cosine_similarity_before is not None:
            payload["cosine_similarity_before"] = self.cosine_similarity_before
        if self.cosine_similarity_after is not None:
            payload["cosine_similarity_after"] = self.cosine_similarity_after
        if self.cosine_similarity_series_after is not None:
            payload["cosine_similarity_series_after"] = self.cosine_similarity_series_after
        return payload

    @classmethod
    def from_dict(cls, data: dict, media_root: Path) -> "Recording":
        return cls(
            id=data["id"],
            name=data["name"],
            kind=RecordingKind(data["kind"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            file_path=media_root / data["file_path"],
            cosine_similarity_before=data.get("cosine_similarity_before"),
            cosine_similarity_after=data.get("cosine_similarity_after"),
            cosine_similarity_series_after=data.get("cosine_similarity_series_after"),
        )


__all__ = [
    "RecordingKind",
    "Speaker",
    "Recording",
]


class ExtractionResult:
    def __init__(
        self,
        id: str,
        mixture_recording_id: str,
        target_speaker_ids: list[str],
        output_recording_ids: list[str],
        processing_ms: float,
        cosine_similarity_before: float | None = None,
        cosine_similarity_after: float | None = None,
    ):
        self.id = id
        self.mixture_recording_id = mixture_recording_id
        self.target_speaker_ids = target_speaker_ids
        self.output_recording_ids = output_recording_ids
        self.processing_ms = processing_ms
        self.cosine_similarity_before = cosine_similarity_before
        self.cosine_similarity_after = cosine_similarity_after

    def to_dict(self) -> dict:
        payload = {
            "id": self.id,
            "mixture_recording_id": self.mixture_recording_id,
            "target_speaker_ids": self.target_speaker_ids,
            "output_recording_ids": self.output_recording_ids,
            "processing_ms": self.processing_ms,
        }
        if self.cosine_similarity_before is not None:
            payload["cosine_similarity_before"] = self.cosine_similarity_before
        if self.cosine_similarity_after is not None:
            payload["cosine_similarity_after"] = self.cosine_similarity_after
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionResult":
        return cls(
            id=data["id"],
            mixture_recording_id=data["mixture_recording_id"],
            target_speaker_ids=list(data["target_speaker_ids"]),
            output_recording_ids=list(data["output_recording_ids"]),
            processing_ms=float(data["processing_ms"]),
            cosine_similarity_before=data.get("cosine_similarity_before"),
            cosine_similarity_after=data.get("cosine_similarity_after"),
        )


__all__.append("ExtractionResult")
