from locust import HttpUser, task, between
import time
import wave
import io
import random

PROMPTS = [
    {
        "category": "Short", 
        "text": "روزی روزگاری در جنگلی مه آلود، قهرمان داستان ما چشم باز کرد."
    },
    {
        "category": "Medium", 
        "text": "پادشاه با صدای بلند فریاد زد: تا زمانی که شمشیر افسانه‌ای را پیدا نکرده‌اید، حق بازگشت به این قلعه را ندارید!"
    },
    {
        "category": "Long", 
        "text": "در شبی تاریک و طوفانی، صدای زوزه گرگ‌ها از دور به گوش می‌رسید. مسافر خسته، کوله‌بارش را روی زمین گذاشت و به شعله‌های لرزان آتش خیره شد. او می‌دانست که فردا روز سرنوشت‌سازی خواهد بود و باید برای زنده ماندن بجنگد."
    }
]

class VoiceAgentTester(HttpUser):
    wait_time = between(2, 5)

    @task
    def test_tts_generation(self):
        prompt = random.choice(PROMPTS)
        payload = {
            "text": prompt["text"],
            "ref_audio_name": "NO_FILE.wav"
        }

        start_time = time.time()
        
        endpoint_name = f"TTS_Generate_{prompt['category']}"

        with self.client.post("/api/tts/generate/", json=payload, name=endpoint_name, catch_response=True) as response:
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    with wave.open(io.BytesIO(response.content), 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        audio_duration = frames / float(rate)
                        
                    rtf = processing_time / audio_duration if audio_duration > 0 else 0
                    
                    print(f"[{prompt['category']}] Proc: {processing_time:.2f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.2f}")

                    if rtf > 1.0:
                        response.failure(f"RTF Limit Exceeded: {rtf:.2f}")
                    else:
                        response.success()

                except Exception as e:
                    response.failure(f"Audio parsing failed: {e}")
            else:
                response.failure(f"HTTP Error {response.status_code}")
