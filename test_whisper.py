import whisper
import os

path = "C:/Users/m/OneDrive/Desktop/Projects/Voice_Diary/dataset/Actor_02/03-01-01-01-01-01-02.wav"

print("Step 1: Loading model...")
model = whisper.load_model("base")

print("Step 2: Checking file...")
print("Exists:", os.path.exists(path))

print("Step 3: Transcribing...")
result = model.transcribe(path)

print("Step 4: Full result:")
print(result)

print("Step 5: Extracted text:")
print("TEXT:", result.get("text"))