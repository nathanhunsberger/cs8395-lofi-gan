import csv
import wave
from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import read
import numpy as np
from scipy.signal import convolve, gaussian


def read_wav(file_path, target_amplitude=50, window_size=1, sigma=4):
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


def divide_into_intervals(signal, framerate, interval_duration):
    interval_size = int(interval_duration * framerate)
    intervals = [signal[i:i + interval_size] for i in range(0, len(signal), interval_size)]
    return intervals


import numpy as np


# def find_max_frequencies(interval, framerate, freq_range=(100, 5000)):
#     n = len(interval)
#     fft_result = np.fft.fft(interval)
#     frequencies = np.fft.fftfreq(n, d=1 / framerate)
#     positive_frequencies = frequencies[:n // 2]
#     magnitude = np.abs(fft_result[:n // 2])
#
#     # Find the indices of the frequencies within the specified range
#     valid_indices = np.where((positive_frequencies >= freq_range[0]) & (positive_frequencies <= freq_range[1]))[0]
#
#     # Find the indices of the top 4 frequencies within the valid range
#     top_indices = valid_indices[np.argsort(magnitude[valid_indices])[-10:]]
#     top_frequencies = positive_frequencies[top_indices]
#     top_amplitudes = magnitude[top_indices]
#
#     return top_frequencies, top_amplitudes

def calculate_frequency_boundaries(start_frequency, end_frequency, num_sections):
    # Calculate the range in octaves
    total_octaves = np.log2(end_frequency / start_frequency)

    # Calculate the width of each section in octaves
    section_width_octaves = total_octaves / num_sections

    # Calculate the boundaries of each section
    boundaries = [start_frequency * 2 ** (i * section_width_octaves) for i in range(num_sections + 1)]

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


regions = calculate_frequency_boundaries(100, 5000, 128)


def find_max_frequencies(interval, framerate):
    n = len(interval)
    fft_result = np.fft.fft(interval)
    frequencies = np.fft.fftfreq(n, d=1 / framerate)
    positive_frequencies = frequencies[:n // 2]
    magnitude = np.abs(fft_result[:n // 2])

    # Sum the amplitudes over each region
    region_sums = np.zeros(len(regions))
    for i, (start, end) in enumerate(regions):
        region_indices = np.where((positive_frequencies >= start) & (positive_frequencies < end))[0]
        region_sums[i] = np.sum(magnitude[region_indices]) / (end - start)

    # Find the indices of the top 10 regions
    top_indices = np.argsort(region_sums)[-10:]
    top_regions = [(regions[i][0] + regions[i][1])/2 for i in top_indices]

    return top_regions, region_sums[top_indices]


from pydub import AudioSegment
from pydub.generators import Sine


def write_results(intervals, framerate, interval_duration):
    # Create an empty audio segment
    output_audio = AudioSegment.silent(duration=1000 * interval_duration * len(intervals))

    filename = "output.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Interval', 'Frequency', 'Volume'])

        for i, interval in enumerate(intervals):
            top_frequencies, top_amplitudes = find_max_frequencies(interval, framerate)
            # print(f"Interval {i + 1}:")
            csv_row = [i]
            csv_row.extend(top_frequencies)
            csv_row.extend(top_amplitudes)
            csvwriter.writerow(csv_row)


file_path = 'mojito.wav'
signal, framerate = read_wav(file_path)

# Set the desired interval duration in seconds
interval_duration = 0.2

intervals = divide_into_intervals(signal, framerate, interval_duration)
write_results(intervals, framerate, interval_duration)
