import os
import torch
import numpy as np
import speech_recognition as sr
import pyttsx3
import sounddevice as sd
from scipy.signal import resample
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import google.generativeai as genai
import time
from gtts import gTTS
import pygame
import tempfile


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "models/gemini-1.5-flash-latest"

MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME).eval()
EMOTIONS = emotion_model.config.id2label

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
voice_dict = {voice.name: voice.id for voice in voices}


def record_audio(duration=3, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return np.squeeze(audio).astype(np.float32)

def preprocess(audio, sr_orig=16000):
    return resample(audio, int(len(audio) * 16000 / sr_orig)).astype(np.float32)

def predict_emotion(audio):
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    return EMOTIONS[torch.argmax(logits).item()]

def speech_to_text():
    with sr.Microphone(sample_rate=16000) as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=1)
        try:
            print(" Listening...")
            audio = recognizer.listen(mic, timeout=5, phrase_time_limit=6)
            return recognizer.recognize_google(audio, language="hi-IN")
        except sr.WaitTimeoutError:
            print(" Timeout: No speech detected.")
            return None
        except Exception as e:
            print("Speech Recognition Error:", e)
            return None

def get_reply(text, emotion):
    try:
        prompt = f"""
        Tum ek pyari, samajhdar aur empathetic hospital assistant ho. 
        Jo har kisi se hindi aur english mix (Hinglish) mein pyaar se baat karti hai.
        Yahan user ne bola (emotion: {emotion}): {text}
        Tumhara jawab helpful, calm,non judgemental,empathetic,natural aur insaan ki feelings ko samajhne wala hona chahiye aur koi emojies mat use karna।
        """
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini API Error:", e)
        return "माफ़ कीजिए, मैं अभी उत्तर देने में असमर्थ हूँ।"


def speak(text, lang='hi', tld='co.in'):
    try:
        tts = gTTS(text=text, lang=lang, tld=tld)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file_path = temp_file.name
        temp_file.close()

        tts.save(temp_file_path)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.3)
        pygame.mixer.quit()

        os.remove(temp_file_path)
    except Exception as e:
        print(" TTS Error:", e)

class VoiceEmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Assistant")

        self.voice_label = ttk.Label(root, text="Choose Voice:")
        self.voice_label.pack(pady=5)

        self.voice_combo = ttk.Combobox(root, values=list(voice_dict.keys()))
        self.voice_combo.current(0)
        self.voice_combo.pack(pady=5)


        self.lang_label = ttk.Label(root, text="Choose TTS Language/Accent:")
        self.lang_label.pack(pady=5)

        self.lang_options = {
            "Hindi Female (hi, com)": ('hi', 'com'),
            "Indian English Accent (en, co.in)": ('en', 'co.in'),
        }
        self.lang_combo = ttk.Combobox(root, values=list(self.lang_options.keys()))
        self.lang_combo.current(0)
        self.lang_combo.pack(pady=5)

        self.start_btn = ttk.Button(root, text=" Start Assistant", command=self.start_assistant)
        self.start_btn.pack(pady=5)

        self.stop_btn = ttk.Button(root, text=" Stop", command=self.stop_assistant)
        self.stop_btn.pack(pady=5)

        ttk.Label(root, text="🗣️ Recognized Speech:").pack()
        self.user_text_box = scrolledtext.ScrolledText(root, height=5, width=60)
        self.user_text_box.pack(padx=10, pady=5)

        ttk.Label(root, text=" Assistant Reply:").pack()
        self.assistant_text_box = scrolledtext.ScrolledText(root, height=7, width=60)
        self.assistant_text_box.pack(padx=10, pady=5)

        ttk.Label(root, text=" Emotion & Logs:").pack()
        self.log_box = scrolledtext.ScrolledText(root, height=5, width=60)
        self.log_box.pack(padx=10, pady=5)

        self.is_running = False

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)

    def update_user_text(self, message):
        self.user_text_box.delete('1.0', tk.END)
        self.user_text_box.insert(tk.END, message)

    def update_assistant_text(self, message):
        self.assistant_text_box.delete('1.0', tk.END)
        self.assistant_text_box.insert(tk.END, message)

    def start_assistant(self):
        if self.is_running:
            self.log(" Already running.")
            return

        voice_name = self.voice_combo.get()
        tts_engine.setProperty('voice', voice_dict[voice_name])
        self.is_running = True
        self.log(" Assistant started.")

        thread = threading.Thread(target=self.run_loop)
        thread.daemon = True
        thread.start()

    def stop_assistant(self):
        self.is_running = False
        self.log("Assistant stopped.")

    def run_loop(self):
        while self.is_running:
            try:
                audio = record_audio()
                emotion = predict_emotion(preprocess(audio))
                self.log(f" Emotion detected: {emotion}")

                text = speech_to_text()
                if not text:
                    self.log("⚠ Could not understand speech.")
                    continue

                self.update_user_text(text)
                self.log(f" Recognized Speech: {text}")

                reply = get_reply(text, emotion)
                self.update_assistant_text(reply)
                self.log(f" Gemini Reply: {reply}")

                # Get selected language/accent for TTS
                lang_key = self.lang_combo.get()
                lang, tld = self.lang_options.get(lang_key, ('hi', 'co.in'))
                speak(reply, lang=lang, tld=tld)

                time.sleep(5)
            except Exception as e:
                self.log(f" Error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceEmotionGUI(root)
    root.mainloop()
