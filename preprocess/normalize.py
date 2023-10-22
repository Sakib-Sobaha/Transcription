from pydub import AudioSegment, effects
import numpy as np
import os


def normalize_audio(file_path):
    # Load the audio file
    # file_path = "/mnt/d/Synesis/ASR/mywav_reduced_noise.wav"
    input_audio = AudioSegment.from_file(file_path)

    # Define the target LUFS loudness level (adjust as needed)
    target_lufs = -16.0  # Adjust according to your preference

    # Calculate the difference between the target and current LUFS level
    lufs_difference = target_lufs - input_audio.dBFS

    # Normalize the audio to the target LUFS level
    normalized_audio = input_audio + lufs_difference

    normalized_audio = effects.normalize(normalized_audio) 

    # Export the processed audio to a new file
    # output_file = "output_audio.wav"
    # path_to_save = os.path.join('./temp', 'denoised_' + filename)
    normalized_audio.export(file_path, format="wav")

    # Print the original and target LUFS levels
    print(f"Original LUFS Level: {input_audio.dBFS:.2f} dB")
    print(f"Target LUFS Level: {target_lufs:.2f} dB")
    print(f"Normalized LUFS Level: {normalized_audio.dBFS:.2f} dB")
    print(f"Normalized audio saved as '{file_path}'")

    return file_path

