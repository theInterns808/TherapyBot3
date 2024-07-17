import os
import asyncio
import sys
import json
import requests
import pickle
import wave
import shutil

sys.path.append('/home/anrric/Desktop/TherapyMultiModalModel/emotion_recognition_using_speech')
from emotion_recognition_using_speech.data_extractor import AudioExtractor
from emotion import detect_emotion
from emotion3 import record_audio, EmotionRecognizer
from datetime import datetime

proxies = {'https': 'http://127.0.0.1:8888'}

def format_sound_emotion(emotion_array):
    return f"Sound Emotion Detected : {emotion_array.tolist()}"

def request(prompt, url="http://localhost:11434/api/generate", emotionFacial=None, emotionSound=None):
    data = {
        "model": "llama3",
        "prompt": f"{prompt} my face's emotion is: {emotionFacial}, my voice's emotion is: {emotionSound}" if emotionFacial and emotionSound else prompt,
        "stream": True
    }
    try:
        response = requests.post(url, json=data, proxies=proxies, stream=True)
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response content: {response.content}")
            return None

        if 'application/x-ndjson' not in response.headers.get('Content-Type', ''):
            print("Error: Response is not in NDJSON format")
            return None

        print("<BraveMind>: ", end="")
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    response_json = json.loads(line.decode('utf-8'))
                    generated_text = response_json.get('response', '')
                    print(generated_text, end="", flush=True)
                    full_response += generated_text

                    if response_json.get('done', False):
                        break

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    return None
        return full_response
        print("\n")
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def cycle(prompt, is_initial_prompt=False):
    detected_sound = record_audio()
    emotionSound = None
    if detected_sound.any():
        try:
            with open("/home/anrric/Desktop/TherapyMultiModalModel/speech_model.pkl", "rb") as model_file:
                loaded_model = pickle.load(model_file)
            
            with wave.open("/tmp/test_recording999.wav", 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(detected_sound)
            
            emotionSound = loaded_model.predict("/tmp/test_recording999.wav")
        except Exception as e:
            print(f"Error processing sound emotion: {e}")
    
    detected_emotion = detect_emotion()
    if is_initial_prompt:
        prompt = f'''Then, You're a therapist for Veterans. Limit your words to less than 20 words. Please base your diagnosis with this emotion. Ask questions to further diagnose and be very conversational when needed.'''
    else:
        prompt = "You are a therapist. " + prompt
    if not detected_emotion:
        print("no emotion detected")
    
    response = request(prompt, emotionFacial=detected_emotion, emotionSound=emotionSound)
    
    if detected_sound.any() and emotionSound:
        try:
            emotion_sound_directory = f"/home/anrric/Desktop/TherapyMultiModalModel/emotion_recognition_using_speech/data/train-custom/{emotionSound}"
            os.makedirs(emotion_sound_directory, exist_ok=True)
            new_path = os.path.join(emotion_sound_directory, f"test_{emotionSound}_{datetime.now()}.wav")
            shutil.move("/tmp/test_recording999.wav", new_path)        
            
            # Save prompt and chat log to a text file named after the emotion
            log_path = os.path.join(emotion_sound_directory, f"{emotionSound}_{datetime.now()}.txt")
            with open(log_path, "a") as log_file:
                log_file.write(f"Prompt: {prompt}\n")
                log_file.write(f"Chat Log: {response}\n\n")
        except Exception as e:
            print(f"Error saving sound emotion data: {e}")
    
    return prompt  

def summarize_prompt(prompt):
    return prompt.split(",")[0]

if __name__ == "__main__":
    try:
        initial_prompt = "Initial prompt to set the context."
        last_response = cycle(initial_prompt, is_initial_prompt=True)
        while True:
            print("\n")
            user_prompt = input("<User>: ")
            last_response = cycle(user_prompt)
    except KeyboardInterrupt:
        summary = summarize_prompt(last_response)
        cycle('''Summarize the conversation before into one word of the list that I will provide for you that explains the whole conversation (happy, sad, fear, angry)''')
        with open(f"/home/anrric/Desktop/TherapyMultiModalModel/emotion_recognition_using_speech/data/train-custom/summary_{summary}.txt", "w") as file:
            file.write(last_response)
        print(f"\n summary_{summary}.txt")
