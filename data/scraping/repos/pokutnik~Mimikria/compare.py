#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import wave
from envoy import run
import numpy as np
import matplotlib.pyplot as mp
from mlab import cohere
import json

def to_wav(filename):
    wavname = filename + ".wav"
    subpr = run('ffmpeg -y -i %s -ac 1 -ar 8000 %s' % (filename, wavname))
    if subpr.status_code == 0:
        return wavname

def load_wav(wavname):
    return wave.open(wavname)

def doit(filename):
    wav = load_wav(to_wav(filename))
    raw = wav.readframes(wav.getnframes())
    a = np.fromstring(raw, dtype='int16')
    return a

def main(orig, user):
    a = doit(orig)
    b = doit(user)
    c, _ = cohere(a,b)
    delay = int(np.argmax(c))
    score = c[delay]
    print json.dumps(dict(score=score, delay=delay))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        a = doit('bf420f5720c41682011ddf05744b1950')
        b = doit('This_is_-_SPARTA.mp3')
        #l = np.correlate(a_sh, b_sh)
        #l = np.correlate(a, b)
        #p = mp.plot(l)
        #mp.cohere(a, b, 256, 1./8000.0)
        c, _ = cohere(a,b)
        delay = int(np.argmax(c))
        score = float(c[delay])
        print delay, score



    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
