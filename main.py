from qai_hub_models.models.whisper_base_en.model import WhisperBaseEn
from qai_hub_models.models._shared.whisper.app import WhisperApp

AUDIO_FILE_PATH = "records/enregistrement_20250624_145939.wav"

def main():
    model = WhisperBaseEn.from_pretrained()
    app = WhisperApp(
        model.encoder,
        model.decoder,
        num_decoder_blocks=model.num_decoder_blocks,
        num_decoder_heads=model.num_decoder_heads,
        attention_dim=model.attention_dim,
        mean_decode_len=model.mean_decode_len,
    )

    # Load default audio if file not provided
    audio = AUDIO_FILE_PATH
    audio_sample_rate = None

    # Perform transcription
    transcription = app.transcribe(audio, audio_sample_rate)
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()