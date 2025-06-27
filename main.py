from stt import SpeechToTextApplication
from time import perf_counter

AUDIO_DIR_PATH = "records"


def main():
    print("Loading SpeechToTextApplication...")
    stt_app = SpeechToTextApplication(AUDIO_DIR_PATH)
    print("SpeechToTextApplication loaded successfully.")

    print("Transcribing audio file...")
    # Perform transcription
    start = perf_counter()
    transcription = stt_app.transcribe()
    end = perf_counter()
    print(f"Transcription({end - start}): {transcription}")
    print("Transcription completed.")

if __name__ == "__main__":
    main()