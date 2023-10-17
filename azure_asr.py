
import time
import azure.cognitiveservices.speech as speechsdk

done = False

speech_key, service_region = "e6f545b48cbc4ccb96c7397335a6560e", "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language="bn-IN"


def azure_asr(filepath):
    audio_config = speechsdk.audio.AudioConfig(filename=filepath)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    global done

    def stop_cb(evt):
        # print('CLOSING on {}'.format(evt))
        speech_recognizer.stop_continuous_recognition()
        global done
        done = True

    azure_list = []

    speech_recognizer.recognized.connect(lambda evt: azure_list.append(evt.result.text))


    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    speech_recognizer.start_continuous_recognition()

    while not done:
        time.sleep(.5)

    azure_text = ''
    for i in azure_list:
        azure_text = azure_text + i

    done = False
    
    return azure_text