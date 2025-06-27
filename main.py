from stt import SpeechToTextApplication
from time import perf_counter

AUDIO_DIR_PATH = "records"


def main():
    stt_app = SpeechToTextApplication(AUDIO_DIR_PATH)

    # Perform transcription
    start = perf_counter()
    transcription = stt_app.transcribe()
    end = perf_counter()
    print(f"Transcription({end - start}): {transcription}")


if __name__ == "__main__":
    main()