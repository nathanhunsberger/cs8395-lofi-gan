import numpy as np
from pydub import AudioSegment
from pydub.generators import Square


def find_max_frequencies(interval, framerate, freq_range=(100, 5000)):
    n = len(interval)
    fft_result = np.fft.fft(interval)
    frequencies = np.fft.fftfreq(n, d=1 / framerate)

    positive_frequencies = frequencies[:n // 2]
    magnitude = np.abs(fft_result[:n // 2])

    # Find the indices of the frequencies within the specified range
    valid_indices = np.where((positive_frequencies >= freq_range[0]) & (positive_frequencies <= freq_range[1]))[0]

    # Find the indices of the top 4 frequencies within the valid range
    top_indices = valid_indices[np.argsort(magnitude[valid_indices])[-4:]]
    top_frequencies = positive_frequencies[top_indices]
    top_amplitudes = magnitude[top_indices]

    return top_frequencies, top_amplitudes


def build_audio_with_overlapping_intervals(input_audio, interval_duration, overlap_ratio=0.5):
    framerate = input_audio.frame_rate
    interval_size = int(interval_duration * 1000)  # convert to milliseconds
    overlap_size = int(interval_size * overlap_ratio)

    output_audio = AudioSegment.silent(duration=10*1000)

    for i in range(0, len(input_audio) - interval_size, overlap_size):
        interval = input_audio[i:i + interval_size]

        # Your frequency analysis function
        frequencies, volumes = find_max_frequencies(interval.get_array_of_samples(), framerate)

        for freq, amp in sorted(zip(frequencies, volumes)):
            sine_wave = Square(freq)
            sine_wave = sine_wave.to_audio_segment(duration=interval_duration * 1000)
            sine_wave = sine_wave.apply_gain(-60 + amp / max(volumes) * 50)

            offset = 0
            desired_segment = sine_wave[offset:offset + int(interval_duration * 1000)]
            output_audio = output_audio.overlay(desired_segment, position=i)

    return output_audio


# Example usage
input_file_path = "lofi-simple.wav"
output_file_path = "nate-sexy.wav"

input_audio = AudioSegment.from_file(input_file_path)
output_audio = build_audio_with_overlapping_intervals(input_audio, interval_duration=0.5, overlap_ratio=0.3)
output_audio.export(output_file_path, format="wav")
