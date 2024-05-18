import os
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext, messagebox
from llmware.models import ModelCatalog
from llmware.gguf_configs import GGUFConfigs
from llmware.setup import Setup
from llmware.exceptions import DependencyNotInstalledException

# Initialize the TTS engine
engine = pyttsx3.init()


def get_available_voices():
    """Get the list of available voices."""
    voices = engine.getProperty('voices')
    available_voices = []
    for voice in voices:
        available_voices.append(voice)
    return available_voices


def set_voice(voice_id):
    """Set the voice (optional)."""
    engine.setProperty('voice', voice_id)


def text_to_speech(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()


def initialize_llmware_model():
    """Initialize and return the LLMware model."""
    GGUFConfigs().set_config("whisper_cpp_verbose", "OFF")
    GGUFConfigs().set_config("whisper_cpp_realtime_display", True)
    GGUFConfigs().set_config("whisper_language", "en")
    GGUFConfigs().set_config("whisper_remove_segment_markers", True)

    # Load the model from ModelCatalog
    whisper_base_english = "whisper-cpp-base-english"
    model = ModelCatalog().load_model(whisper_base_english)
    return model


class ChatApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("AI Chat Box")
        self.model = model

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

        self.update_chat_window(f"You: {user_text}\n")

        try:
            response_text = self.generate_response(user_text)
        except DependencyNotInstalledException as e:
            response_text = f"Error: Missing dependency: {str(e)}. Please install the required packages."

        self.update_chat_window(f"AI Response: {response_text}\n")

        set_voice(self.selected_voice.get())
        text_to_speech(response_text)

    def generate_response(self, user_text):
        """Generate a response from the LLMware model based on user input."""
        # Use the model to generate a response
        response = self.model.inference(user_text)
        return response["llm_response"]

    def update_chat_window(self, text):
        """Update the chat window with the given text."""
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, text)
        self.chat_window.config(state=tk.DISABLED)


if __name__ == "__main__":
    model = initialize_llmware_model()
    root = tk.Tk()
    app = ChatApp(root, model)
    root.mainloop()
