from flask import Flask, send_file
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import time
from pydub import AudioSegment  # MP3 dönüştürme için pydub kullanıyoruz

app = Flask(__name__)

@app.route('/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)

    conditioning = [{
        "prompt": "128 BPM tech house drum loop",
        "seconds_start": 0, 
        "seconds_total": 30
    }]
    
    # Audio generate etme
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )
    
    # Audio verisini işleme
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # Dinamik dosya adı
    timestamp = int(time.time())
    wav_filename = f"output_{timestamp}.wav"
    mp3_filename = f"output_{timestamp}.mp3"
    
    # WAV dosyasını kaydet
    torchaudio.save(wav_filename, output, sample_rate)
    
    # WAV dosyasını MP3'e dönüştür
    audio = AudioSegment.from_wav(wav_filename)
    audio.export(mp3_filename, format="mp3")
    
    # MP3 dosyasını gönder
    return send_file(mp3_filename, mimetype='audio/mp3')

if __name__ == '__main__':
    app.run(debug=True)