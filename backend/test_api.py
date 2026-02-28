import requests
import numpy as np
import soundfile as sf
import io

# Create a dummy audio file (sine wave)
sr = 22050
t = np.linspace(0, 3, sr * 3)
y = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave

# Save to in-memory buffer
buffer = io.BytesIO()
sf.write(buffer, y, sr, format='WAV')
buffer.seek(0)

# Send request
url = "http://localhost:8000/predict"
files = {'file': ('test.wav', buffer, 'audio/wav')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
