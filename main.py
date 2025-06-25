from whisper import SpeechToTextApplication


AUDIO_DIR_PATH = "records"


def main():
    stt_app = SpeechToTextApplication(AUDIO_DIR_PATH)

    # Perform transcription
    transcription = stt_app.transcribe()
    print("Transcription:", transcription)


if __name__ == "__main__":
    main()