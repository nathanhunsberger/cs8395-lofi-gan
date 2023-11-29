import csv
from pydub import AudioSegment
from pydub.generators import Sine, Square


def build_audio(interval_duration):
    filename = "output.csv"
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        num_rows = len(list(reader))
    # Create an empty audio segment
    output_audio = AudioSegment.silent(duration=1000 * interval_duration * num_rows)

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row

        i = 1
        offset = 0
        for row in csvreader:
            frequencies = [float(x) for x in row[1:4]]
            volumes = [float(x) for x in row[4:]]

            for freq, amp in zip(frequencies, volumes):
                sine_wave = Sine(freq)
                sine_wave = sine_wave.to_audio_segment(duration=1000 * interval_duration)
                sine_wave = sine_wave - (60 - amp / 15000 * 25)  # Adjust volume

                # offset = (offset + 47) % 100  # Beautiful random number generator
                desired_segment = sine_wave[offset:offset + 1000 * interval_duration]
                output_audio = output_audio.overlay(desired_segment, position=i * 1000 * interval_duration)

                # output_audio = output_audio.overlay(sine_wave, position=i * 1000 * interval_duration)
                # print(f"  Frequency: {freq} Hz, Amplitude: {amp}")
            i +=1

    # Increase the overall volume
    volume_scaling = 0  # Adjust as needed
    output_audio = output_audio + volume_scaling

    # Export the resulting audio to a WAV file
    output_file_path = 'test_construct.wav'
    output_audio.export(output_file_path, format='wav')


# Set the desired interval duration in seconds
interval_duration = 0.2

build_audio(interval_duration)
