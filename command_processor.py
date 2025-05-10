from google import genai
import tuya_executor
import json
import re
import transcriptor
import pyttsx3
import time

from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env

def text_to_speech(text):
    print(text)
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

chat = client.chats.create(model="gemini-2.0-flash")

def response(query):
    response = chat.send_message(query)
    return response.text

def generate(spatial_information):
    chat.send_message(
        "You are an IoT Device Controller. Your task is to execute user commands based on natural language input."
        "Take decisions based on the user's natural language"
        "For instance if the user completes studying, turn off the lights that are used for studying - probably near a desk."
        "Another instance: if the user wants to sleep, turn off all the lights and turn on the fan."
        "Respond with only the necessary device commands in the format of a dictionary with key value pairs as  'deviceName: On' or 'deviceName: Off'.\n"
        "1. If the user indicates they are leaving the room, turn off all devices.\n"
        "2. If the user is present in the room, intelligently turn on devices that would be useful for them.\n"
        "4. Provide only the commands without additional explanations.\n\n"
        "The spatial information for your context is as follows: " + spatial_information
    )
    comm = ''
    while True:
        comm = transcriptor.transcribe()
        #print("User: ",comm)
        if comm=='exit':
            break
        res = response(comm)
        print(res)
        json_match = re.search(r'```json\n(.*?)\n```', res, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
            data = json.loads(json_string)
            print(data)  # Output: {'Light1': 'On', 'Light3': 'On', 'Light4': 'On'}
        else:
            data = eval(res)
        for device,command in data.items():
            text_to_speech(f'Turning {command} the device {device}')
            tuya_executor.execute(device,command)
        print("Tell your next command in 3 seconds..")
        for i in range(3):
            time.sleep(1)
            print(3-i,"....")
