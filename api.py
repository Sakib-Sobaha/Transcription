import time
import datetime

import numpy as np
import torch
import uvicorn
import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

from postprocess import postprocess_text

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
    result = model.transcribe(audio_data)['text']

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
    # result = result.replace('ï¿½', '')
    result = postprocess_text(result)

    time_taken = time.time() - start_time
    # data = {
    #     "time_taken": time_taken,
    #     "result": result
    # }

    print("At:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ", Chars:", len(result), ", Time_taken:", round(time_taken, 3))
    data = {
        "taskType": "synesis-asr",
        "output": [{"source": result}],
        "char_count": len(result),
        "time_taken": time_taken,
    }

    return data


if __name__ == "__main__":
    model = whisper.load_model("small.en", device='cuda')
    model_bn = pipeline('automatic-speech-recognition',
                    model="sazzad-sit/whisper-small-bn-3ds", max_new_tokens=448, \
                           device=0, batch_size=16, chunk_length_s=25)
    uvicorn.run(app, host="0.0.0.0", port=9852)