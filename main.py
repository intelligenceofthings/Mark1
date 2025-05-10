from onboarder import onboarding_system
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env

IMAGE_PATH = os.getenv("IMAGE_PATH")

if not Path("spatial_information.txt").exists():
    spatial = onboarding_system(IMAGE_PATH)
else:
    with open("spatial_information.txt", "r") as file:
        spatial = file.read()

import command_processor
command_processor.generate(spatial)
