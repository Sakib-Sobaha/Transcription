import os

def find_audio_files(directory):
    audio_files = []

    for root, dirs, files in os.walk(directory):
        # Check if we've gone two layers deep
        if root.startswith(directory) and root[len(directory):].count(os.sep) <= 3:
            for file in files:
                if file.lower().endswith((".wav", ".mp3", ".csv")):
                    audio_files.append(os.path.join(root, file))

    return audio_files