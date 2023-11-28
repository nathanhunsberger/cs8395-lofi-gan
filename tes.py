import numpy as np

def calculate_frequency_boundaries(start_frequency, end_frequency, num_sections):
    # Calculate the range in octaves
    total_octaves = np.log2(end_frequency / start_frequency)

    # Calculate the width of each section in octaves
    section_width_octaves = total_octaves / num_sections

    # Calculate the boundaries of each section
    boundaries = [start_frequency * 2**(i * section_width_octaves) for i in range(num_sections + 1)]

    return boundaries

# Example usage for a range from 100 Hz to 2000 Hz with 10 sections
start_frequency = 50
end_frequency = 4000
num_sections = 64

frequency_boundaries = calculate_frequency_boundaries(start_frequency, end_frequency, num_sections)

# Print the results
for i, boundary in enumerate(frequency_boundaries[:-1]):
    print(f"Section {i + 1}: {boundary:.2f} Hz to {frequency_boundaries[i + 1]:.2f} Hz")
