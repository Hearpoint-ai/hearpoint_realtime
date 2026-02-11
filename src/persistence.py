import json
from pathlib import Path
from typing import Iterable, Tuple

from .models import ExtractionResult, Recording, Speaker


class MediaJsonStore:
    """Lightweight persistence layer that can be swapped out for a database later."""

    def __init__(self, media_root: Path, data_file: Path):
        self.media_root = media_root
        self.data_file = data_file

    def load(self) -> Tuple[list[Speaker], list[Recording], list[ExtractionResult]]:
        if not self.data_file.exists():
            return [], [], []
        payload = json.loads(self.data_file.read_text())
        speakers = [Speaker.from_dict(item, self.media_root) for item in payload.get("speakers", [])]
        recordings = [Recording.from_dict(item, self.media_root) for item in payload.get("recordings", [])]
        extractions = [ExtractionResult.from_dict(item) for item in payload.get("extractions", [])]
        return speakers, recordings, extractions

    def save(
        self,
        speakers: Iterable[Speaker],
        recordings: Iterable[Recording],
        extractions: Iterable[ExtractionResult],
    ) -> None:
        payload = {
            "speakers": [speaker.to_dict(self.media_root) for speaker in speakers],
            "recordings": [recording.to_dict(self.media_root) for recording in recordings],
            "extractions": [extraction.to_dict() for extraction in extractions],
        }
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.data_file.write_text(json.dumps(payload, indent=2))
