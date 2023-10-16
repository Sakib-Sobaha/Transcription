import torch
torch.set_num_threads(1)

# from pprint import pprint
import os

SAMPLING_RATE = 16000

def vad(model, utils, audio_path, filename):
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
    # pprint(speech_timestamps)

    path_to_save = os.path.join('./temp', 'vad_' + filename)

    save_audio(path_to_save,
           collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 

    return path_to_save