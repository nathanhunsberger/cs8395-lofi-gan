import wave
from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import read
import numpy as np
from scipy.signal import convolve, gaussian

def read_wav(file_path, target_amplitude=50, window_size=20, sigma=4):
    framerate, signal = read(file_path)

    # If the signal is stereo, take only one channel (convert to mono)
    if len(signal.shape) == 2:
        signal = signal[:, 0]

    # Find the maximum amplitude in the entire audio signal
    max_amplitude = np.max(np.abs(signal))

    # Normalize the entire audio signal based on the maximum amplitude
    normalized_signal = (signal / max_amplitude) * target_amplitude

    # Create a Gaussian window
    window = gaussian(window_size, std=sigma)
    print(window)
    normalized_window = window / np.sum(window)

    # Apply convolution with the Gaussian window
    smoothed_signal = convolve(normalized_signal, normalized_window, mode='same')

    return smoothed_signal, framerate
# def read_wav(file_path, target_amplitude=10):
#     framerate, signal = read(file_path)
#
#     # If the signal is stereo, take only one channel (convert to mono)
#     if len(signal.shape) == 2:
#         signal = signal[:, 0]
#
#     # Find the maximum amplitude in the entire audio signal
#     max_amplitude = np.max(np.abs(signal))
#
#     # Normalize the entire audio signal based on the maximum amplitude
#     normalized_signal = (signal / max_amplitude) * target_amplitude
#
#     return normalized_signal, framerate

# def read_wav(file_path):
#     framerate, signal = read(file_path)
#
#     # If the signal is stereo, take only one channel (convert to mono)
#     if len(signal.shape) == 2:
#         signal = signal[:, 0]
#
#     return signal, framerate

# def read_wav(file_path):
#     with wave.open(file_path, 'rb') as wf:
#         framerate = wf.getframerate()
#         frames = wf.readframes(wf.getnframes())
#         signal = [int.from_bytes(frames[i:i+2], byteorder='little', signed=True) for i in range(0, len(frames), 2)]
#     return signal, framerate
def divide_into_intervals(signal, framerate, interval_duration):
    interval_size = int(interval_duration * framerate)
    intervals = [signal[i:i+interval_size] for i in range(0, len(signal), interval_size)]
    return intervals
import numpy as np

def find_max_frequencies(interval, framerate, freq_range=(200, 1200)):
    n = len(interval)
    fft_result = np.fft.fft(interval)
    frequencies = np.fft.fftfreq(n, d=1/framerate)
    positive_frequencies = frequencies[:n//2]
    magnitude = np.abs(fft_result[:n//2])

    # Find the indices of the frequencies within the specified range
    valid_indices = np.where((positive_frequencies >= freq_range[0]) & (positive_frequencies <= freq_range[1]))[0]

    # Find the indices of the top 4 frequencies within the valid range
    top_indices = valid_indices[np.argsort(magnitude[valid_indices])[-10:]]
    top_frequencies = positive_frequencies[top_indices]
    top_amplitudes = magnitude[top_indices]

    return top_frequencies, top_amplitudes

from pydub import AudioSegment
from pydub.generators import Sine

def print_results(intervals, framerate, interval_duration):
    # Create an empty audio segment
    output_audio = AudioSegment.silent(duration=1000*interval_duration * len(intervals))

    for i, interval in enumerate(intervals):
        top_frequencies, top_amplitudes = find_max_frequencies(interval, framerate)
        # print(f"Interval {i + 1}:")
        count = 10
        for freq, amp in zip(top_frequencies, top_amplitudes):
            sine_wave = Sine(freq)
            sine_wave = sine_wave.to_audio_segment(duration=1000*interval_duration)
            sine_wave = sine_wave - (60 - amp/15000 * 25)  # Adjust volume
            count -= 2
            output_audio = output_audio.overlay(sine_wave, position=i * 1000 * interval_duration)
            # print(f"  Frequency: {freq} Hz, Amplitude: {amp}")

    # Increase the overall volume
    volume_scaling = 0  # Adjust as needed
    output_audio = output_audio + volume_scaling

    # Export the resulting audio to a WAV file
    output_file_path = 'test_construct.wav'
    output_audio.export(output_file_path, format='wav')
file_path = 'mojito.wav'
signal, framerate = read_wav(file_path)

# Set the desired interval duration in seconds
interval_duration = 0.2

intervals = divide_into_intervals(signal, framerate, interval_duration)
print_results(intervals, framerate, interval_duration)
