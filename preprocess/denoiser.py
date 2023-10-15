import os
import torch
from df import enhance
from df.enhance import load_audio, save_audio
# from df.enhance import enhance, init_df, load_audio, save_audio

# model, df_state, _ = init_df()


# https://github.com/Rikorose/DeepFilterNet

# def denoise(model, df_state, audio):

#     with sf.SoundFile(audio) as audio_file:
#         sample_rate = audio_file.samplerate

#     audio_chunks = [audio[:, i:i + 60 * sample_rate]
#                     for i in range(0, audio.shape[1], 60 * sample_rate)]

#     enhanced_chunks = []
#     for ac in audio_chunks:
#         enhanced_chunks.append(enhance(model, df_state, ac))

#     enhanced_audio = torch.cat(enhanced_chunks, dim=1)

#     return enhanced_audio


def denoise(model, df_state, file_path, filename):
    audio, info = load_audio(file_path, sr=df_state.sr())

    # Split audio into 10min chunks
    audio_chunks = [audio[:, i:i + 60 * info.sample_rate]
                    for i in range(0, audio.shape[1], 60 * info.sample_rate)]

    enhanced_chunks = []
    for ac in audio_chunks:
        enhanced_chunks.append(enhance(model, df_state, ac))

    enhanced = torch.cat(enhanced_chunks, dim=1)

    # assert enhanced.shape == audio.shape, 'Enhanced audio shape does not match original audio shape.'

    path_to_save = os.path.join('./temp', 'denoised_' + filename)

    save_audio(path_to_save, enhanced, sr=df_state.sr())

    return path_to_save