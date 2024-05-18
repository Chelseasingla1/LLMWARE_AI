import os
import pyttsx3
from llmware.models import ModelCatalog
from llmware.gguf_configs import GGUFConfigs
from llmware.setup import Setup

# Initialize the TTS engine
engine = pyttsx3.init()

def get_available_voices():
    """Get the list of available voices"""
    voices = engine.getProperty('voices')
    for voice in voices:
        print("Voice:")
        print(" - ID: %s" % voice.id)
        print(" - Name: %s" % voice.name)
        print(" - Languages: %s" % voice.languages)
        print(" - Gender: %s" % voice.gender)
        print(" - Age: %s" % voice.age)
        print(" ")

def set_voice(voice_id):
    """Set the voice (optional)"""
    engine.setProperty('voice', voice_id)

def text_to_speech(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def sample_files(example="famous_quotes", small_only=False, use_tts=True):
    """Execute a basic inference on Voice-to-Text model passing a file_path string"""
    GGUFConfigs().set_config("whisper_cpp_verbose", "OFF")
    GGUFConfigs().set_config("whisper_cpp_realtime_display", True)
    GGUFConfigs().set_config("whisper_language", "en")
    GGUFConfigs().set_config("whisper_remove_segment_markers", True)

    voice_samples = Setup().load_voice_sample_files(small_only=small_only)

    examples = ["famous_quotes", "greatest_speeches", "youtube_demos", "earnings_calls"]

    if example not in examples:
        print("choose one of the following - ", examples)
        return 0

    fp = os.path.join(voice_samples, example)

    files = os.listdir(fp)

    whisper_base_english = "whisper-cpp-base-english"
    model = ModelCatalog().load_model(whisper_base_english)

    for f in files:
        if f.endswith(".wav"):
            prompt = os.path.join(fp, f)
            print(f"\n\nPROCESSING: prompt = {prompt}")
            response = model.inference(prompt)
            print("\nllm response: ", response["llm_response"])
            print("usage: ", response["usage"])

            if use_tts:
                # Convert response to speech
                text_to_speech(response["llm_response"])

    return 0

if __name__ == "__main__":
    get_available_voices()
    voice_id = input("Enter the voice ID you want to use (or press Enter to use the default voice): ")
    if voice_id:
        set_voice(voice_id)

    use_tts = input("Do you want to use text-to-speech? (yes/no): ")
    if use_tts.lower() == 'yes':
        sample_files(example="famous_quotes", small_only=False, use_tts=True)
    else:
        sample_files(example="famous_quotes", small_only=False, use_tts=False)