import os
import pyttsx3
import streamlit as st
from llmware.models import ModelCatalog
from llmware.gguf_configs import GGUFConfigs
from llmware.setup import Setup

# Initialize the TTS engine
engine = pyttsx3.init()

def get_available_voices():
    """Get the list of available voices"""
    voices = engine.getProperty('voices')
    voice_options = []
    for voice in voices:
        voice_options.append((voice.name, voice.id))
    return voice_options

def set_voice(voice_id):
    """Set the voice (optional)"""
    engine.setProperty('voice', voice_id)

def text_to_speech(text):
    """Convert text to speech"""
    print(f"Text-to-Speech: {text}")  # Debug message
    engine.say(text)
    engine.runAndWait()

def simple_chat_ui_app(model_name, voice_id):
    # Set the selected voice
    set_voice(voice_id)

    st.title(f"Simple Chat with {model_name}")

    GGUFConfigs().set_config("max_output_tokens", 500)
    model = ModelCatalog().load_model(model_name, temperature=0.3, sample=True, max_output=450)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    prompt = st.chat_input("Say something")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Note that the st.write_stream method consumes a generator - so pass model.stream(prompt) directly
            bot_response = "".join([chunk for chunk in model.stream(prompt)])
            st.markdown(bot_response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Convert bot response to speech
        text_to_speech(bot_response)

    return 0

if __name__ == "__main__":
    # Get available voices
    voice_options = get_available_voices()
    voice_names = [voice[0] for voice in voice_options]
    voice_ids = {voice[0]: voice[1] for voice in voice_options}

    # Streamlit sidebar for voice selection
    st.sidebar.title("Settings")
    selected_voice_name = st.sidebar.selectbox("Select Voice", voice_names)
    selected_voice_id = voice_ids[selected_voice_name]

    # A few representative good chat models that can run locally
    chat_models = [
        "phi-3-gguf",
        "llama-2-7b-chat-gguf",
        "llama-3-instruct-bartowski-gguf",
        "openhermes-mistral-7b-gguf",
        "zephyr-7b-gguf",
        "tiny-llama-chat-gguf"
    ]

    model_name = st.sidebar.selectbox("Select Model", chat_models)

    simple_chat_ui_app(model_name, selected_voice_id)
