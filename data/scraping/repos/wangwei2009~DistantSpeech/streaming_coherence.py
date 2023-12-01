

import time
import pyaudio
import numpy as np
from beamformer.MicArray import MicArray
from coherence.BinauralEnhancement import BinauralEnhancement
from beamformer.utils import mesh,pmesh,load_wav

from beamformer.realtime_processing import realtime_processing

if __name__ == "__main__":

    r = 0.032
    c = 343

    frameLen = 256
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    c = 343
    r = 0.032*2
    fs = 16000

    MicArray = MicArray(arrayType='linear', r=r, c=c, M=2)
    angle = np.array([270, 0]) / 180 * np.pi

    dualMIC_Enhancement = BinauralEnhancement(MicArray, frameLen, hop, nfft, c, r, fs)
    # yout = fixedbeamformer.superDirectiveMVDR2(x,angle)

    rec = realtime_processing(EnhancementMehtod=dualMIC_Enhancement,chunk=4096, angle=angle)
    rec.audioDevice()
    print("Start processing...\n")
    rec.start()
    while True:
        a = int(input('"select algorithm: \n'
                      '0.src  \n'
                      '1.coherence:endfire  \n'
                      '2.coherence:broadside \n'
                      '3.coherence:endfire complex-m1 \n'
                      '4.coherence:endfire complex-m3 \n'
                      '5.coherence:endfire Competing-Talker \n'))
        rec.changeAlgorithm(a)
        # time.sleep(0.1)