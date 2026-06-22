from rag.ingest import build_rag_index

import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import tempfile
import os
import sqlite3
import numpy as np



build_rag_index([
    "knowledge/appointment_notes.txt",
    "knowledge/medications.txt",
    "knowledge/faq.txt",
])
_whisper_model = None


class User:

    def __init__(
        self,
        user_id,
        name,
        semantic_path=None,
        steering_path=None
    ):
        self.user_id = user_id
        self.name = name
        self.semantic_path = semantic_path
        self.steering_path = steering_path



def get_whisper():
    global _whisper_model

    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")

    return _whisper_model


def record_and_append_to_file(
    filename: str,
    duration: int = 10,
):

    fs = 16000

    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="int16",
    )

    sd.wait()

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as tmp:

        write(
            tmp.name,
            fs,
            audio,
        )

        wav_path = tmp.name

    model = get_whisper()

    result = model.transcribe(wav_path)

    transcript = result["text"]

    with open(
        filename,
        "a",
        encoding="utf-8",
    ) as f:

        f.write("\n")
        f.write(transcript.strip())
        f.write("\n")

    os.remove(wav_path)

    return transcript