import sys
import os
import torch
from django.apps import AppConfig
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

class TtsApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api_chatterbox'
    
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def ready(self):
        if os.environ.get('RUN_MAIN') == 'true' or 'gunicorn' in sys.argv[0]:
            self.load_model()

    def load_model(self):
        print("\n==========================================")
        print(f"[TTS SERVER] Loading Custom Persian Chatterbox securely onto {self.device}...")
        print("==========================================\n")
        
        if "./chatterbox_git/src" not in sys.path:
            sys.path.append("./chatterbox_git/src")
            
        from chatterbox import mtl_tts
        import chatterbox

        try:
            #downloading requirments
            model_dir = snapshot_download(repo_id="Thomcles/Chatterbox-TTS-Persian-Farsi")
            print(f"[INFO] Downloaded/Found model files at: {model_dir}")
            
            # load base model
            multilingual_model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=TtsApiConfig.device)
            
            # baching t3 model
            t3_path = os.path.join(model_dir, "t3_fa.safetensors")
            if os.path.exists(t3_path):
                t3_state = load_safetensors(t3_path, device=TtsApiConfig.device)
                multilingual_model.t3.load_state_dict(t3_state, strict=False)
                multilingual_model.t3.to(TtsApiConfig.device).eval()
                print("[INFO] Persian t3 weights loaded.")
            
            s3gen_path = os.path.join(model_dir, "s3gen_fa.safetensors") 
            if os.path.exists(s3gen_path):
                s3gen_state = load_safetensors(s3gen_path, device=TtsApiConfig.device)
                multilingual_model.s3gen.load_state_dict(s3gen_state, strict=False)
                multilingual_model.s3gen.to(TtsApiConfig.device).eval()
                print("[INFO] Persian s3gen weights loaded.")
            else:
                model_safetensors = os.path.join(model_dir, "model.safetensors")
                if os.path.exists(model_safetensors):
                    try:
                        state = load_safetensors(model_safetensors, device=TtsApiConfig.device)
                        multilingual_model.s3gen.load_state_dict(state, strict=False)
                        print("[INFO] s3gen patched from main model.safetensors")
                    except:
                        pass
                        
            TtsApiConfig.model = multilingual_model
            vram = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"\n [SUCCESS] Model Patched & Loaded! VRAM: {vram:.2f} MB\n")
            
        except Exception as e:
            print(f"\n [ERROR] Failed to load model: {e}\n")