from groq import Groq

from dotenv import load_dotenv
import os

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def parse_command(client,user_input):
    pre_prompt = "Provide the output strictly as a dictionary with object names as keys and their counts as values. For example, if the input is '2 lights, one fan,' the output should be: {'light': 2, 'fan': 1}. Do not include any preamble or additional text."
    prompt = pre_prompt + user_input
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    return response.choices[0].message.content

def refine(command):
    client = Groq(api_key=GROQ_API_KEY)
    obj_dic = parse_command(client,command)
    return obj_dic
