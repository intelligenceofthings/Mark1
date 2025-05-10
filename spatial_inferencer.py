import os
from openai import OpenAI
import base64
from dotenv import load_dotenv
import os

load_dotenv() 
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

token = OPENAI_KEY
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

image_path = './annotated_image.jpg'

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def information(object_list):
    base64_image = encode_image(image_path)
    prompt = (
    f"Analyze the provided image and the annotations {object_list}, ensuring the response does not exceed 1000 words. "
    "Summarize the spatial placement of each annotated IoT device with the following key details:\n\n"
    
    "**1. Object Location:**\n"
    "   - Briefly describe each device's position relative to major room features (e.g., walls, windows, doors, furniture).\n"
    
    "**2. Nearby Objects:**\n"
    "   - Identify the closest objects (IoT and non-IoT) and summarize their influence on placement.\n"
    
    "**3. Spatial Relationships:**\n"
    "   - Note relative depth, alignment, and positioning concisely, prioritizing only the most relevant details.\n"
    
    "Keep descriptions brief, precise, and within the word limit while maintaining clarity."
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
                
            }
        ],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model_name
    )

    return response.choices[0].message.content


    
    