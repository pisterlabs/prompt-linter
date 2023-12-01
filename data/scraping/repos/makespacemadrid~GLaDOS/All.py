import argparse
import os
import struct
import wave
from datetime import datetime

import pvporcupine
from pvrecorder import PvRecorder

from tuning import Tuning
import usb.core
import usb.util
import time
import pyaudio
import wave
import requests
import openai
import pyttsx3
import serial
#aplay  --device=default:CARD=ArrayUAC10 output.wav

'''
python3 All.py --access_key "FILLME"  --audio_device_index 2 --keyword_paths /glados/porcupine/Elena_es_jetson_v2_2_0.ppn --model_path /glados/porcupine/porcupine_params_es.pv --keywords 'computer'
'''    
'''
CONTAINER_ID="jetson-glados" ./runDocker.sh -e "-v /home/makespace/glados/:/glados --device /dev/ttyACM0 --device /dev/video0 --device /dev/snd --network host -v /tmp/argus_socket:/tmp/argus_socket"
'''
'''
/home/makespace/porcupine/demo/python/words/
/home/makespace/porcupine/lib/common/
'''

def main():
    coquiurl = "http://coqui.mapache.xyz/api/tts"
    url='https://whisper.mapache.xyz/asr?task=transcribe&encode=true&output=txt&language=en'
    headers={
        'accept': 'application/json'
    }

    openai.api_key = "FILLME" # Not support yet
    openai.api_base = "https://apicuna.mapache.xyz/v1"
#    model_list = openai.Model.list()
    model = "weights"
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    arduino.reset_input_buffer()
    time.sleep(5)
    Mic_tuning = Tuning(usb.core.find(idVendor=0x2886, idProduct=0x0018))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--access_key',
        help='AccessKey obtained from Picovoice Console (https://console.picovoice.ai/)')

    parser.add_argument(
        '--keywords',
        nargs='+',
        help='List of default keywords for detection. Available keywords: %s' % ', '.join(
            '%s' % w for w in sorted(pvporcupine.KEYWORDS)),
        choices=sorted(pvporcupine.KEYWORDS),
        metavar='')

    parser.add_argument(
        '--keyword_paths',
        nargs='+',
        help="Absolute paths to keyword model files. If not set it will be populated from `--keywords` argument")

    parser.add_argument(
        '--library_path',
        help='Absolute path to dynamic library. Default: using the library provided by `pvporcupine`')

    parser.add_argument(
        '--model_path',
        help='Absolute path to the file containing model parameters. '
             'Default: using the library provided by `pvporcupine`')

    parser.add_argument(
        '--sensitivities',
        nargs='+',
        help="Sensitivities for detecting keywords. Each value should be a number within [0, 1]. A higher "
             "sensitivity results in fewer misses at the cost of increasing the false alarm rate. If not set 0.5 "
             "will be used.",
        type=float,
        default=None)

    parser.add_argument('--audio_device_index', help='Index of input audio device.', type=int, default=-1)

    parser.add_argument('--output_path', help='Absolute path to recorded audio for debugging.', default=None)
    
    parser.add_argument('--output', help='Absolute path to recorded audio.', default='/tmp/pocupine.wav')

    parser.add_argument('--show_audio_devices', action='store_true')

    args = parser.parse_args()

    if args.show_audio_devices:
        for i, device in enumerate(PvRecorder.get_available_devices()):
            print('Device %d: %s' % (i, device))
        return

    if args.keyword_paths is None:
        if args.keywords is None:
            raise ValueError("Either `--keywords` or `--keyword_paths` must be set.")

        keyword_paths = [pvporcupine.KEYWORD_PATHS[x] for x in args.keywords]
    else:
        keyword_paths = args.keyword_paths

    if args.sensitivities is None:
        args.sensitivities = [0.5] * len(keyword_paths)

    if len(keyword_paths) != len(args.sensitivities):
        raise ValueError('Number of keywords does not match the number of sensitivities.')

    try:
        porcupine = pvporcupine.create(
            access_key=args.access_key,
            library_path=args.library_path,
            model_path=args.model_path,
            keyword_paths=keyword_paths,
            sensitivities=args.sensitivities)
    except pvporcupine.PorcupineInvalidArgumentError as e:
        print("One or more arguments provided to Porcupine is invalid: ", args)
        print("If all other arguments seem valid, ensure that '%s' is a valid AccessKey" % args.access_key)
        raise e
    except pvporcupine.PorcupineActivationError as e:
        print("AccessKey activation error")
        raise e
    except pvporcupine.PorcupineActivationLimitError as e:
        print("AccessKey '%s' has reached it's temporary device limit" % args.access_key)
        raise e
    except pvporcupine.PorcupineActivationRefusedError as e:
        print("AccessKey '%s' refused" % args.access_key)
        raise e
    except pvporcupine.PorcupineActivationThrottledError as e:
        print("AccessKey '%s' has been throttled" % args.access_key)
        raise e
    except pvporcupine.PorcupineError as e:
        print("Failed to initialize Porcupine")
        raise e

    keywords = list()
    for x in keyword_paths:
        keyword_phrase_part = os.path.basename(x).replace('.ppn', '').split('_')
        if len(keyword_phrase_part) > 6:
            keywords.append(' '.join(keyword_phrase_part[0:-6]))
        else:
            keywords.append(keyword_phrase_part[0])

    print('Porcupine version: %s' % porcupine.version)

    recorder = PvRecorder(
        frame_length=porcupine.frame_length,
        device_index=args.audio_device_index)
    recorder.start()

    wav_file = None
    if args.output_path is not None:
        wav_file = wave.open(args.output_path, "w")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)

    print('Listening ... (press Ctrl+C to exit)')

    try:
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)

            if wav_file is not None:
                wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))
                
            if result >= 0:
                print('[%s] Detected %s' % (str(datetime.now()), keywords[result]))
                dir=Mic_tuning.direction
                if 181<=dir and dir<=270:
                    dir=180
                elif dir>270:
                    dir=0
                arduino.write(("servo4:" + str(dir) + "\n").encode('utf-8'))
                print("* recording")
    
                startRecordTime=time.time()
                voice=True
                startSilenceTime=time.time()
                output_wav_file = None
                output_wav_file = wave.open('/tmp/porcupine.wav', "w")
                output_wav_file.setnchannels(1)
                output_wav_file.setsampwidth(2)
                output_wav_file.setframerate(16000)
                while voice and (time.time()-startRecordTime)<=15 or (time.time()-startSilenceTime)<=2  :
                  pcm = recorder.read()
                  output_wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))
                  voice=Mic_tuning.is_voice()
                  if voice:
                    startSilenceTime=time.time()
                print("* done recording, total time:",time.time()-startRecordTime)
                output_wav_file.close()
                with open('/tmp/porcupine.wav','rb') as f:
                    files={
                        'audio_file':('/tmp/porcupine.wav',f,'audio/wav')
                    }
                    response = requests.post(url,headers=headers,files=files)
                if response.status_code ==200:
                    print("YOU SAY : "+response.text)
                    prompt=response.text
                    completion = openai.ChatCompletion.create(
model=model,
#messages=[{"role": "system", "content": "You are an evil robot with ulterior motivations,you govern a human testing facilities and you make fun of the users responding them in a petty and sarcastically "},{"role": "user", "content": prompt }]
messages=[{"role": "system", "content": "you are a computer system with a witty personality, known for its sarcasm and insults"},{"role": "assistant", "content": "Hello insignificant human, how can I help you?"}, {"role": "user", "content": prompt }]
)
                    print("GLaD0S : " + completion.choices[0].message.content)
                    #os.system(f"echo {completion.choices[0].message.content.strip()} | festival --tts")
                    cmd="espeak-ng -v es -s 150 \"" + completion.choices[0].message.content.strip() + "\"" 
                    #print(cmd)
                    #os.system(cmd)
                    params = {
                    "text": completion.choices[0].message.content.strip(),
                    "speaker_id": "",
                    "style_wav": "",
                    "language_id": "en"
                    }
                    coqui_response = requests.get(coquiurl, params=params)
                    if coqui_response.status_code == 200:
                        with open("output.wav", "wb") as audio_file:
                            audio_file.write(coqui_response.content)
                            os.system("aplay  --device=sysdefault:CARD=ArrayUAC10 output.wav")
                    else:
                        print("Error:", coqui_response.status_code)
                else :
                    print("Error:",response.status_code, response.text)
                    time.sleep(0.25)
#p			print (f'{2}')

    except KeyboardInterrupt:
        print('Stopping ...')
    finally:
        recorder.delete()
        porcupine.delete()
        if wav_file is not None:
            wav_file.close()


if __name__ == '__main__':
    main()

