from transformers import pipeline
from huggingface_hub import snapshot_download
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-14B"
LOCAL_DIR = os.path.join(BASE_DIR, "../../models/Qwen/Qwen3-14B")

print(LOCAL_DIR)

def load_pipeline():
    try:
        print("Loadin model")
        pipe = pipeline(
            "text-generation",
            model=LOCAL_DIR,
        )
        print("Modelo carregado localmente.")
    except Exception:
        print("Modelo não encontrado localmente. A fazer download...")
        
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        
        pipe = pipeline(
            "text-generation",
            model=LOCAL_DIR
        )
    
    return pipe

# Remove think from model output
def clean_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def use_model():
    pipe = load_pipeline()

    messages = [
        {"role": "user", "content": "Who are you?"}
    ]

    result = pipe(messages)
    result = clean_output(result[0]["generated_text"][-1]["content"])
    print(result)
    
use_model()
