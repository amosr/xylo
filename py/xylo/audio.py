import pyaudio
import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.signal
import wave
import math

import os

format     = pyaudio.paFloat32
rate       = 44100
channels   = 1
num_frames = rate // 2

"""
Read samples from microphone, blocking
"""
def read_samples(dur_s: float):
  p = pyaudio.PyAudio()
  stream = p.open(rate = rate, channels = channels, format = format, input = True)
  try:
    bytes = stream.read(int(dur_s * rate), exception_on_overflow=False)
    return np.frombuffer(bytes, dtype=np.float32)
  finally:
    stream.close()

"""Read wave file from disk"""
def read_wave(fp, prefix='data/wav/'):
  with wave.open(prefix + fp, 'rb') as w:
    assert(w.getframerate() == rate)
    assert(w.getnchannels() == 1)
    assert(w.getsampwidth() == 4)
    nf = w.getnframes()
    bytes = w.readframes(nf)
    i32 = np.frombuffer(bytes, 'int32')
    fs = i32 / math.pow(2, 31)
    return fs

"""Save audio to wave file"""
def write_wave(fp, arr, prefix='data/wav/'):
  fs = arr * math.pow(2, 31)
  i32 = fs.astype('int32')
  count = 0
  fn = f'{prefix}{fp}.wav'
  while os.path.isfile(fn):
    count = count + 1
    fn = f'{prefix}{fp}_{count:03}.wav'

  with wave.open(fn, 'wb') as w:
    w.setnchannels(1)
    w.setframerate(rate)
    w.setsampwidth(4)
    w.writeframes(i32)

def spectrum(arr: np.ndarray, height: float = 30.0, distance: float = 2000, prominence: float = 30.0, take_freqs: int = 40000):
  A = np.fft.rfft(arr * np.hamming(len(arr)))
  Aa = np.abs(A)
  Aa = Aa[0:take_freqs]
  Aa = Aa / np.linalg.norm(Aa)

  (pk,d) = scipy.signal.find_peaks(Aa,  height = height, distance = distance, prominence = prominence)
  return (Aa, pk, d)

def plot_spectrum(arr: np.ndarray, height: float = 0.02, distance: float = 2000, prominence: float = 0.02, take_freqs: int = 40000, comparison_freq: float = 1.0):
  (Aa, pk, d) = spectrum(arr, height, distance, prominence, take_freqs)
  freqs = np.fft.rfftfreq(len(arr), 1 / rate)

  plt.plot(freqs[0:len(Aa)], Aa)

  print(freqs[pk], freqs[pk] / comparison_freq)
  print(d)
  plt.xticks(freqs[pk], freqs[pk].astype('int'))

def print_spectrum(arr: np.ndarray, height: float = 30.0, distance: float = 2000, prominence: float = 30.0, take_freqs: int = 40000, comparison_freq: float = 1.0):
  (Aa, pk, d) = spectrum(arr, height, distance, prominence, take_freqs)
  freqs = np.fft.rfftfreq(len(arr), 1 / rate)

  print(freqs[pk], freqs[pk] / comparison_freq)

def list_spectrum(dir = 'data/wav', **kwargs):
  for fp in sorted(os.listdir(dir)):
    if fp.endswith('.wav'):
      w = read_wave(fp, prefix=dir + '/')
      print(fp)
      print_spectrum(w, **kwargs)


def spectrum_diff(arr1: np.ndarray, arr2: np.ndarray, height: float = 30.0, distance: float = 2000, prominence: float = 30.0, take_freqs: int = 40000, width: float = 1, plot: bool = False):
  (Aa1, pk1, d1) = spectrum(arr1, height, distance, prominence, take_freqs)
  (Aa2, pk2, d2) = spectrum(arr2, height, distance, prominence, take_freqs)

  # Aa1 = Aa1 / np.linalg.norm(Aa1)
  # Aa2 = Aa2 / np.linalg.norm(Aa2)

  Aa  = Aa1 + Aa2

  (pk,d) = scipy.signal.find_peaks(Aa, height = height, distance = distance, prominence = prominence, width = width)

  freqs = np.fft.rfftfreq(len(arr1), 1 / rate)

  if plot:
    plt.plot(freqs[0:len(Aa)], Aa, color = 'red')
    plt.plot(freqs[0:len(Aa)], Aa1, color = 'blue')
    plt.plot(freqs[0:len(Aa)], Aa2, color = 'green')

    print(freqs[pk])
    print(d)
    plt.xticks(freqs[pk], freqs[pk].astype('int'))

  return freqs[pk], d
