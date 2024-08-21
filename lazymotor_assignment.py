import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline
import edge_tts
import asyncio
import os

# Constants
AUDIO_FILENAME = "input.wav"
RESPONSE_FILENAME = "output.mp3"
DURATION = 5  # Duration to record in seconds
SAMPLERATE = 16000

# Step 1: Voice-to-Text Conversion

def record_audio(duration=DURATION, samplerate=SAMPLERATE):
    try:
        print("Recording...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        print("Recording complete.")
        return audio
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None

def save_wav(filename, audio, samplerate):
    try:
        wav.write(filename, samplerate, audio)
        print(f"Audio saved to {filename}")
    except Exception as e:
        print(f"Error saving audio: {e}")

def transcribe_audio(filename):
    try:
        print("Transcribing audio...")
        model = whisper.load_model("base")
        result = model.transcribe(filename)
        if result and 'text' in result:
            text = result['text']
            print("Transcription complete.")
            return text
        else:
            print("Transcription result is missing or invalid.")
            return None
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

# Step 2: Text Input into LLM

def generate_response(text):
    try:
        if text is None:
            raise ValueError("No text provided for response generation.")
        print("Generating response...")
        llm_pipeline = pipeline("text-generation", model="gpt-3.5-turbo")
        result = llm_pipeline(text, max_length=50, num_return_sequences=1)
        if result and len(result) > 0 and 'generated_text' in result[0]:
            response = result[0]['generated_text']
            response = '. '.join(response.split('. ')[:2]) + '.'
            print("Response generation complete.")
            return response
        else:
            print("Response generation result is missing or invalid.")
            return None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Step 3: Text-to-Speech Conversion

async def text_to_speech(text, output_file=RESPONSE_FILENAME, pitch="0", speed="1.0", voice="en-US-JennyNeural"):
    try:
        if text is None:
            raise ValueError("No text provided for text-to-speech.")
        print("Converting text to speech...")
        communicate = edge_tts.Communicate()
        communicate.voice = voice
        communicate.pitch = pitch
        communicate.rate = speed
        await communicate.save_file(output_file, text)
        print(f"Audio response saved as {output_file}")
    except Exception as e:
        print(f"Error converting text to speech: {e}")

def main():
    try:
        # Record and save audio
        audio = record_audio()
        if audio is not None:
            save_wav(AUDIO_FILENAME, audio, SAMPLERATE)

            # Transcribe the saved audio file
            text = transcribe_audio(AUDIO_FILENAME)
            if text is not None:
                print("Transcribed Text:", text)

                # Generate and print response
                response_text = generate_response(text)
                if response_text is not None:
                    print("Response Text:", response_text)

                    # Convert text to speech
                    asyncio.run(text_to_speech(response_text))
                else:
                    print("Failed to generate response.")
            else:
                print("Failed to transcribe audio.")
        else:
            print("Failed to record audio.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
