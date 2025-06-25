from qai_hub_models.models.whisper_base_en.model import WhisperBaseEn
from qai_hub_models.models._shared.whisper.app import WhisperApp
from pathlib import Path

# TODO: Add type hints to the methods and class attributes
# TODO: Add docstrings to the methods and class attributes
# TODO: Add streaming support to the transcribe method


class SpeechToTextApplication:
    def __init__(self, audio_records_path: Path | str | None = None):
        self.model = WhisperBaseEn.from_pretrained()
        self.app = WhisperApp(
            self.model.encoder,
            self.model.decoder,
            num_decoder_blocks=self.model.num_decoder_blocks,
            num_decoder_heads=self.model.num_decoder_heads,
            attention_dim=self.model.attention_dim,
            mean_decode_len=self.model.mean_decode_len,
        )
        self.audio_records_path = Path(audio_records_path) if type(audio_records_path) == str else audio_records_path
        self.last_audio_file = None

    def _get_audio_file(self) -> str:
        records_dir = self.audio_records_path
        audio_files = list(records_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError("No audio files found.")
        self.last_audio_file = audio_files[0]
        return str(audio_files[0])

    def _delete_audio_file(self):
        if self.last_audio_file and Path(self.last_audio_file).exists():
            Path(self.last_audio_file).unlink()
            print(f"Deleted audio file: {self.last_audio_file}")
            self.last_audio_file = None
        else:
            print("No audio file to delete or file does not exist.")

    def transcribe(self, audio: Path | str) -> str:
        if self.audio_records_path is None:
            raise ValueError("Audio records path is not set.")
    
        if isinstance(audio, (str, Path)):
            audio = Path(audio)
            if not audio.exists():
                raise FileNotFoundError(f"Audio file {audio} does not exist.")

        transcription = self.app.transcribe(audio, audio_sample_rate=None)
        print(f"Transcription result: {transcription}")
        self._delete_audio_file()
        return transcription

        
        

        