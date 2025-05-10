import os
import json
import wave
import pyaudio
from groq import Groq

from dotenv import load_dotenv
import os

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper prefers 16kHz audio
CHUNK = 1024
RECORD_SECONDS = 5  # Adjust this for longer recordings
WAVE_OUTPUT_FILENAME = "real_time_audio.wav"

def record_audio():
    """Records audio from the microphone and saves it as a WAV file."""
    audio = pyaudio.PyAudio()

    # Open audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording... Speak now!")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording complete!")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return WAVE_OUTPUT_FILENAME


def transcribe():
    audio_file = record_audio()
    # Initialize the Groq client
    client = Groq(api_key=GROQ_API_KEY)
    # Specify the path to the audio file
    filename = "./real_time_audio.wav" # Replace with your audio file!
    # Open the audio file
    with open(filename, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
        file=file, # Required audio file
        model="whisper-large-v3", # Required model to use for transcription
        prompt="Specify context or spelling. this usually contains the count of electronic devices. Translate the audio to english if needed.",  # Optional
        response_format="verbose_json",  # Optional
        timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
        language="en",  # Optional
        temperature=0.0  # Optional
        )
        # To print only the transcription text, you'd use print(transcription.text) (here we're printing the entire transcription object to access timestamps)
        print("USER: ",transcription.text)
        return transcription.text
    