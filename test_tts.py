from news_pipeline import TextToSpeech
from pathlib import Path
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the TTS engine
output_dir = Path("./news_output")
tts = TextToSpeech(output_dir, client)

# Test text
test_text = "This is a test of the OpenAI text to speech functionality. It should generate an audio file quickly and reliably."

# Convert text to speech
audio_file = tts.convert_text_to_speech(test_text, "test_audio")

if audio_file:
    print(f"Audio file generated: {audio_file}")
else:
    print("Failed to generate audio file.")
