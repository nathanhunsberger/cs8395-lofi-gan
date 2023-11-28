import csv
import math

from pydub import AudioSegment
from pydub.generators import Sine, Square
import numpy as np
import random


# Function to generate a sine wave with a given frequency, amplitude, and phase shift
# def generate_sine_wave(duration, frequency, amplitude, phase_shift):
#     t = np.arange(0, duration / 1000.0, 1 / 44100.0)  # Assuming a sample rate of 44.1 kHz
#     wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift) / 10e7
#     return AudioSegment(wave.tobytes(), frame_rate=44100, sample_width=2, channels=1)


def build_audio(interval_duration):
    filename = "output.csv"
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        num_rows = len(list(reader))
    # Create an empty audio segment
    output_audio = AudioSegment.silent(duration=1000 * interval_duration * (num_rows + 1))

    num_freq = 10
    max_vol = 0
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row

        for row in csvreader:
            max_vol = max(max_vol, max([float(x) for x in row[1 + num_freq:]]))

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row

        i = 1
        output_audios = [AudioSegment.silent(duration=interval_duration)] * num_freq
        offset = 12
        for row in csvreader:

            frequencies = [float(x) for x in row[1:1 + num_freq]]
            volumes = [float(x) for x in row[1 + num_freq:]]
            # print(len(frequencies))
            # print(len(output_audios))
            j = 0
            for freq, amp in sorted(zip(frequencies, volumes)):
                # print(amp)
                #
                sine_wave = Square(freq)
                sine_wave = sine_wave.to_audio_segment(duration=1000 * interval_duration + 100)
                sine_wave = sine_wave.apply_gain(-60 + amp / max_vol * 50)

                offset = (offset + 47) % 100 # Beautiful random number generator
                desired_segment = sine_wave[offset:offset+1000 * interval_duration]
                output_audio = output_audio.overlay(desired_segment, position=i * 1000 * interval_duration)


                # output_audios[j] = output_audios[j].append(
                #     generate_sine_wave(1000 * interval_duration, freq, amp / max_vol / 100000, random.uniform(0, 2 * np.pi)),
                #     crossfade=int(interval_duration / 4))
                j += 1
                # print(f"  Frequency: {freq} Hz, Amplitude: {amp}")
            i += 1
        for output in output_audios:
            output_audio = output_audio.overlay(output)
    # Increase the overall volume
    volume_scaling = 0  # Adjust as needed
    output_audio = output_audio + volume_scaling

    # Export the resulting audio to a WAV file
    output_file_path = 'test_smile.wav'
    output_audio.export(output_file_path, format='wav')


# Set the desired interval duration in seconds
interval_duration = 0.2

build_audio(interval_duration)
