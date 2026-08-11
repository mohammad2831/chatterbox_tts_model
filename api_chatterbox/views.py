import os
import re
import torch
import torchaudio as ta
import traceback
from pathlib import Path

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse

from .apps import TtsApiConfig
import threading
#use lock for handel reqquests
tts_lock = threading.Lock()

def clean_persian_text(text):
    cleaned = re.sub(r'[^\u0600-\u06FF\s0-9.,!?]', '', text)
    cleaned = cleaned.replace('\u200c', ' ')
    cleaned = ' '.join(cleaned.split())
    return cleaned


import re

SENTENCE_SPLITTER = re.compile(r'([.?!؟!؛\n]+)')
COMMA_SPLITTER = re.compile(r'(،+)')
PUNCT_ONLY_CHECK = re.compile(r'^[\s.?!؟!؛،\n]+$')

def chunk_persian_text(text, max_words=15):
    if not text:
        return []
        
    final_chunks = []
    parts = SENTENCE_SPLITTER.split(text)
    sentences = []
    current_chunk = ""
    
    for part in parts:
        if SENTENCE_SPLITTER.match(part):
            current_chunk += part
            if current_chunk.strip():
                sentences.append(current_chunk.strip())
            current_chunk = ""
        else:
            current_chunk += part
            
    if current_chunk.strip():
        sentences.append(current_chunk.strip())

    for sentence in sentences:
        if len(sentence.split()) <= max_words:
            final_chunks.append(sentence)
        else:
            sub_parts = COMMA_SPLITTER.split(sentence)
            current_sub = ""
            for sub in sub_parts:
                if COMMA_SPLITTER.match(sub):
                    current_sub += sub
                    if current_sub.strip():
                        final_chunks.append(current_sub.strip())
                    current_sub = ""
                else:
                    current_sub += sub
            if current_sub.strip():
                final_chunks.append(current_sub.strip())

    return [chunk for chunk in final_chunks if not PUNCT_ONLY_CHECK.match(chunk)]


class GenerateTTSView(APIView):
    def post(self, request):
        print("###################################", flush=True)
        raw_text = request.data.get('text')
        ref_audio_name = request.data.get('ref_audio_name', 'sample_ref.wav')

        temperature = float(request.data.get('temperature', 0.7))
        cfg_weight = float(request.data.get('cfg_weight', 0.5))
        top_p = float(request.data.get('top_p', 0.5))
        exaggeration = float(request.data.get('exaggeration', 0.6))

        if not raw_text:
            return Response({"error": "shold send text"}, status=status.HTTP_400_BAD_REQUEST)

        # clening input text
        target_text = clean_persian_text(raw_text)
        
        if not target_text:
            return Response({"error": "error from cleaning"}, status=status.HTTP_400_BAD_REQUEST)

        if getattr(TtsApiConfig, 'model', None) is None:
            return Response({"error": "model is not loaded"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        

        ref_audio_path = None
        if ref_audio_name and ref_audio_name != "NO_FILE.wav":
            expected_path = os.path.join(base_dir, "audio_samples", ref_audio_name)
            
            if not os.path.exists(expected_path):
                return Response(
                    {"error": f"audio sample '{ref_audio_name}' in {expected_path} not found"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            ref_audio_path = expected_path
            print(ref_audio_path, flush=True)

        try:
            #chunking
            text_chunks = chunk_persian_text(target_text, max_words=18)
            all_waveforms = []
            sample_rate = TtsApiConfig.model.sr

            #lock the proces
            with tts_lock:
                with torch.inference_mode():
                    for idx, chunk in enumerate(text_chunks):
                        print(f"Processing chunk {idx+1}/{len(text_chunks)}: {chunk}", flush=True)
                        
                        wav_chunk = TtsApiConfig.model.generate(
                            text=chunk,
                            language_id=None,
                            temperature=temperature,
                            cfg_weight=cfg_weight,
                            top_p=top_p,
                            exaggeration=exaggeration,
                            audio_prompt_path=ref_audio_path
                        )
                        
                        if isinstance(wav_chunk, torch.Tensor):
                            wav_chunk = wav_chunk.cpu().to(torch.float32)
                            
                            
                            wav_chunk = wav_chunk.view(1, -1)
                            
                            all_waveforms.append(wav_chunk)
                            
                            if idx < len(text_chunks) - 1:
                                silence_tensor = torch.zeros((1, int(sample_rate * 0.3)), dtype=torch.float32)
                                all_waveforms.append(silence_tensor)

            if not all_waveforms:
                return Response({"error": "audio is not generated "}, status=status.HTTP_400_BAD_REQUEST)

            #combination of audio files
            final_waveform = torch.cat(all_waveforms, dim=1)

            desired_speed = float(request.data.get('speed', 0.85))
            
            if desired_speed != 1.0:
                effects = [
                    ['tempo', str(desired_speed)]
                ]
                final_waveform, sample_rate = ta.sox_effects.apply_effects_tensor(final_waveform, sample_rate, effects)
            # -----------------------------------

        
            output_dir = Path(base_dir) / "media" / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"output_{hash(target_text)}.wav"

        
            max_amp = final_waveform.abs().max()
            if max_amp > 1.0:
                final_waveform = final_waveform / max_amp

            ta.save(
                str(output_path), 
                final_waveform, 
                sample_rate,
                encoding="PCM_S", 
                bits_per_sample=16
            )

            response = FileResponse(open(output_path, 'rb'), content_type='audio/wav')
            return response

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e), "trace": traceback.format_exc()}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




