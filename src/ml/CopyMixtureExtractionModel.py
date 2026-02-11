import shutil
from pathlib import Path
from typing import List

from .interfaces import TargetSpeechExtractionModel


class CopyMixtureExtractionModel(TargetSpeechExtractionModel):
    def separate(
        self,
        mixture_audio_path: Path,
        speaker_embedding_paths: List[Path],
        output_dir: Path,
        output_name_prefix: str,
    ) -> List[Path]:
        outputs: List[Path] = []
        for index, _ in enumerate(speaker_embedding_paths):
            output_path = output_dir / f"{output_name_prefix}_{index}.wav"
            shutil.copyfile(mixture_audio_path, output_path)
            outputs.append(output_path)
        return outputs
