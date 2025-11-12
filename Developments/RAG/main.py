import dotenv
import os
from utils.utils import VibeThinker


try:
    os.getcwd()
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print('Error|dotenv module not found. Please install it using "pip install python-dotenv".')

model_path = "WeiboAI/VibeThinker-1.5B"
model = VibeThinker(model_path)

if __name__ == '__main__':
    prompt = 'Hello i am Mateo. What is my name?'
    print(model.infer_text(prompt))