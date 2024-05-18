import os
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from llmware.models import ModelCatalog
from llmware.gguf_configs import GGUFConfigs
from llmware.setup import Setup

# Initialize the TTS engine
engine = pyttsx3.init()

def get_available_voices():
    """Get the list of available voices"""
    voices = engine.getProperty('voices')
    available_voices = []
    for voice in voices:
        available_voices.append(voice)
        print("Voice:")
        print(" - ID: %s" % voice.id)
        print(" - Name: %s" % voice.name)
        print(" - Languages: %s" % voice.languages)
        print(" - Gender: %s" % voice.gender)
        print(" - Age: %s" % voice.age)
        print(" ")
    return available_voices

def set_voice(voice_id):
    """Set the voice (optional)"""
    engine.setProperty('voice', voice_id)

def text_to_speech(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def sample_files(example="famous_quotes", small_only=False):
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

            # Convert response to speech
            text_to_speech(response["llm_response"])

    return 0

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chat Box")

        self.chat_window = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=20)
        self.chat_window.grid(row=0, column=0, padx=10, pady=10)
        self.chat_window.config(state=tk.DISABLED)

        self.user_input = tk.Entry(self.root, width=40)
        self.user_input.grid(row=1, column=0, padx=10, pady=10)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.voices = get_available_voices()
        self.selected_voice = tk.StringVar(self.root)
        self.selected_voice.set(self.voices[0].id)  # Set default voice
        self.voice_menu = tk.OptionMenu(self.root, self.selected_voice, *[voice.id for voice in self.voices])
        self.voice_menu.grid(row=2, column=0, padx=10, pady=10)

    def send_message(self):
        user_text = self.user_input.get()
        if user_text.strip() == "":
            messagebox.showwarning("Warning", "You must enter a message.")
            return

        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, "You: " + user_text + "\n")
        self.chat_window.config(state=tk.DISABLED)
        self.user_input.delete(0, tk.END)

        if user_text.strip().lower() == "yes":
            response_text = "Available voices:\n"
            for voice in self.voices:
                response_text += f"- ID: {voice.id}, Name: {voice.name}, Languages: {voice.languages}, Gender: {voice.gender}, Age: {voice.age}\n"
        else:
            # Simulate AI response for demonstration
            response_text = "AI Response: This is a response from the AI."

        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, response_text + "\n")
        self.chat_window.config(state=tk.DISABLED)

        set_voice(self.selected_voice.get())
        text_to_speech(response_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
