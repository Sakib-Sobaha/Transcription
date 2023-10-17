import os
import io
import time

import numpy as np
import torch
import soundfile as sf
import uvicorn
import whisper
from loguru import logger
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from df import init_df


from postprocess import postprocess_text
from preprocess.denoiser import denoise
from preprocess.vad import vad
from azure_asr import azure_asr


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return "Welcome to Synesis ASR API Services."


@app.post("/en/")
async def english_transcription(sound: UploadFile = File(...)):
    start_time = time.time()

    contents = await sound.read()
    audio_data = torch.from_numpy(np.frombuffer(
        contents, np.int16).flatten().astype(np.float32) / 32768.0)
    
    data_size = audio_data.size()[0]
    result = model_en.transcribe(audio_data)['text']

    time_taken = time.time() - start_time
    data = {
        "size": data_size,
        "time_taken": time_taken,
        "result": result
    }
    return data



@app.post("/bn/")
async def bengali_transcription(sound: UploadFile = File(...)):
    start_time = time.time()

    contents = await sound.read()
    result = model_bn(contents)['text']
    result = postprocess_text(result)

    time_taken = time.time() - start_time

    # print("At:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ", Chars:", len(result), ", Time_taken:", round(time_taken, 3))
    logger.success(f"Chars: {len(result)}, Time_taken: {round(time_taken, 3)}")
    data = {
        "taskType": "synesis-asr",
        "output": [{"source": result}],
        "char_count": len(result),
        "time_taken": time_taken,
    }

    return data


@app.post("/bn-enhanced/")
async def bengali_transcription_enhanced(sound: UploadFile = File(...), \
                                            apply_denoiser: bool = False, \
                                            apply_vad: bool = False, \
                                            asr: str = 'whisper'):
    start_time = time.time()

    try:
        contents = await sound.read()  

        # converting byte audio to wav
        bytes_io = io.BytesIO(contents)
        audio_data, sample_rate = sf.read(bytes_io)

        if not os.path.exists('temp'):
            os.makedirs('temp')
        file_path = os.path.join('./temp', sound.filename)
        sf.write(file_path, audio_data, sample_rate)


        if apply_denoiser == True:
            denoised_audio_path = denoise(denoiser_model, denoiser_df_state, file_path, sound.filename)
            file_path = denoised_audio_path
            # print(file_path)


        if apply_vad == True:
            vad_audio_path = vad(vad_model, vad_utils, file_path, sound.filename)
            file_path = vad_audio_path
            # print(file_path)

        if asr.lower() == 'azure':
            result = azure_asr(file_path)
        
        else:
            result = model_bn(file_path)['text']
            result = postprocess_text(result)

        time_taken = round(time.time() - start_time, 3)

        logger.success(f"Chars: {len(result)}, Time_taken: {time_taken}")
        data = {
            "status": 200,
            "taskType": "synesis-asr",
            "output": [{"source": result}],
            "char_count": len(result),
            "time_taken": time_taken,
        }

        return data
    except Exception as e:
        time_taken = time.time() - start_time
        logger.error(f"Failed. Error Message: {e}")
        data = {
            "status": 400,
            "taskType": "synesis-asr",
            "message": str(e),
            "time_taken": time_taken,
        }

        return data

if __name__ == "__main__":
    model_en = whisper.load_model("small.en", device='cuda')
    model_bn = pipeline('automatic-speech-recognition',
                    model="sazzad-sit/whisper-small-bn-3ds", max_new_tokens=448, \
                           device=0, batch_size=16, chunk_length_s=25)
    denoiser_model, denoiser_df_state, _ = init_df()
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

    uvicorn.run(app, host="0.0.0.0", port=9852)