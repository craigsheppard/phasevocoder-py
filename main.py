#!/usr/bin/env python3
"""
Python implementation of SoundHack's Phase Vocoder.
This CLI utility allows time-stretching and pitch-shifting of audio files.
"""

import argparse
import math
import numpy as np
import soundfile as sf
import sys
from scipy import signal

# Constants
PI = math.pi
TWO_PI = 2 * PI

# Window types
HAMMING = 1
VONHANN = 2
KAISER = 3
RAMP = 4
RECTANGLE = 5
SINC = 6
TRIANGLE = 7

# Phase vocoder modes
TIME_MODE = True
PITCH_MODE = False

# FFT directions
TIME2FREQ = 1
FREQ2TIME = 0


class PvocInfo:
    """Stores phase vocoder parameters"""

    def __init__(self):
        self.points = 1024  # FFT size
        self.half_points = self.points // 2
        self.window_size = 4096  # Window size
        self.decimation = 256  # Number of samples to step between analysis frames
        self.interpolation = 256  # Number of samples to step between synthesis frames
        self.window_type = VONHANN
        self.phase_locking = False
        self.scale_factor = 1.0
        self.mask_ratio = 0.0
        self.min_amplitude = 0.0
        self.analysis_type = 0
        self.time = TIME_MODE
        self.use_function = False
        self.overlap = 4  # Overlap factor (window_size = points * overlap)


class SoundInfo:
    """Stores sound file information"""

    def __init__(self):
        self.sample_rate = 44100
        self.n_chans = 1  # Mono by default
        self.frame_size = 4  # Float32 by default
        self.data_start = 0
        self.data_end = 0
        self.num_bytes = 0


def get_window(size, window_type):
    """Create a window of specified type and size"""
    if window_type == HAMMING:
        # Hamming window: a=0.54, b=0.46
        return np.array([0.54 - 0.46 * math.cos(TWO_PI * i / (size - 1)) for i in range(size)])
    elif window_type == VONHANN:
        # Von Hann window: a=0.50, b=0.40
        return np.array([0.50 - 0.40 * math.cos(TWO_PI * i / (size - 1)) for i in range(size)])
    elif window_type == KAISER:
        return np.kaiser(size, 6.8)
    elif window_type == RAMP:
        return np.array([1.0 - (i / size) for i in range(size)])
    elif window_type == RECTANGLE:
        return np.ones(size)
    elif window_type == SINC:
        half_size = size / 2
        window = np.zeros(size)
        for i in range(size):
            if half_size == i:
                window[i] = 1.0
            else:
                window[i] = size * (math.sin(PI * (i - half_size) / half_size) / (2.0 * PI * (i - half_size)))
        return window
    elif window_type == TRIANGLE:
        up = True
        tmp_float = 0.0
        window = np.zeros(size)
        for i in range(size):
            window[i] = 2.0 * tmp_float
            if up:
                tmp_float = tmp_float + 1.0 / size
                if tmp_float > 0.5:
                    tmp_float = 1.0 - tmp_float
                    up = False
            else:
                tmp_float = tmp_float - 1.0 / size
        return window
    else:
        # Default to Hann window
        return np.hanning(size)


def scale_windows(analysis_window, synthesis_window, my_pi):
    """Scale analysis and synthesis windows for unity gain"""
    # When windowSize > points, also apply sin(x)/x windows
    if my_pi.window_size > my_pi.points:
        half_window = -((my_pi.window_size - 1.0) / 2.0)
        for index in range(my_pi.window_size):
            if half_window != 0.0:
                analysis_window[index] = analysis_window[index] * my_pi.points * \
                                         (math.sin(PI * half_window / my_pi.points) / (PI * half_window))
                if my_pi.interpolation:
                    synthesis_window[index] = synthesis_window[index] * my_pi.interpolation * \
                                              (math.sin(PI * half_window / my_pi.interpolation) / (PI * half_window))
            half_window += 1.0

    # Normalize windows for unity gain across unmodified analysis-synthesis procedure
    sum_val = np.sum(analysis_window)
    anal_factor = 2.0 / sum_val
    if my_pi.analysis_type == 0:  # CSOUND_ANALYSIS (assuming 0 is the default)
        anal_factor *= my_pi.decimation * 0.5

    synth_factor = 1.0 / anal_factor if my_pi.window_size > my_pi.points else anal_factor

    analysis_window *= anal_factor
    synthesis_window *= synth_factor

    if my_pi.window_size <= my_pi.points:
        sum_val = 0.0
        for index in range(0, my_pi.window_size, my_pi.interpolation):
            sum_val += synthesis_window[index] * synthesis_window[index]
        sum_val = 1.0 / sum_val
        synthesis_window *= sum_val

    return True


def shift_in(samples, current_position, window_size, decimation, channel_idx=0):
    """Shift next decimation samples into buffer for one channel, returning the number of valid samples read"""
    # Create a window_size buffer
    block = np.zeros(window_size)
    
    # Extract the channel we want if samples has multiple channels
    if len(samples.shape) > 1 and samples.shape[1] > 1:
        channel_samples = samples[:, channel_idx]
    else:
        channel_samples = samples
    
    # When we first start, we don't have enough previous samples to fill the buffer
    if current_position < window_size - decimation:
        # Just read as many samples as we can for the first few frames
        samples_to_read = min(window_size, len(channel_samples) - current_position)
        if samples_to_read <= 0:
            return -2, 0  # End of file
            
        if samples_to_read > 0:
            # Place samples at the end of the block
            offset = window_size - samples_to_read
            block[offset:offset + samples_to_read] = channel_samples[current_position:current_position + samples_to_read]
            return block, samples_to_read
    else:
        # Move remainder from last block into left side of array
        start_pos = current_position - (window_size - decimation)
        # Ensure we don't go below 0
        if start_pos < 0:
            # Adjust how many samples we copy
            valid_samples = window_size - decimation + start_pos
            # Fill first part with zeros if needed
            block[:window_size - decimation - valid_samples] = 0
            # Copy what we can
            if valid_samples > 0:
                block[window_size - decimation - valid_samples:window_size - decimation] = channel_samples[0:valid_samples]
        else:
            # Normal case - we have enough previous samples
            end_pos = min(current_position, len(channel_samples))
            valid_samples = end_pos - start_pos
            block[:valid_samples] = channel_samples[start_pos:end_pos]
            
            # If we couldn't fill the whole buffer, pad with zeros
            if valid_samples < window_size - decimation:
                block[valid_samples:window_size - decimation] = 0
    
    # Read in new samples if available
    remaining = len(channel_samples) - current_position
    
    if remaining <= 0:
        return -2, 0  # Signal end of file
    
    # Read up to decimation samples
    samples_to_read = min(decimation, remaining)
    if samples_to_read > 0:
        block[window_size - decimation:window_size - decimation + samples_to_read] = channel_samples[current_position:current_position + samples_to_read]
    
    # Pad with zeros if we couldn't read enough samples
    if samples_to_read < decimation:
        block[window_size - decimation + samples_to_read:] = 0
        
    return block, samples_to_read


def window_fold(input_block, window, current_time, points, window_size):
    """Apply window to input and fold into output array using modulo arithmetic"""
    output = np.zeros(points)

    # Normalize position
    while current_time < 0:
        current_time += points
    current_time %= points

    # Apply window and fold
    for i in range(window_size):
        output[current_time] += input_block[i] * window[i]
        current_time = (current_time + 1) % points

    return output


def real_fft(data, num_points, direction):
    """Perform real FFT or inverse FFT on data"""
    if direction == TIME2FREQ:
        # Forward FFT - ensure we get exactly num_points//2+1 complex values
        fft_result = np.fft.rfft(data, n=num_points)
        # Convert to SoundHack's interleaved real/imag format
        spectrum = np.zeros(num_points)
        spectrum[0] = np.real(fft_result[0])  # DC component (real)
        spectrum[1] = np.real(fft_result[-1])  # Nyquist component (real)

        for i in range(1, num_points // 2):
            spectrum[i * 2] = np.real(fft_result[i])
            spectrum[i * 2 + 1] = np.imag(fft_result[i])

        return spectrum
    else:
        # Convert from SoundHack's interleaved format to complex values
        half_points = num_points // 2
        complex_data = np.zeros(half_points + 1, dtype=np.complex128)
        complex_data[0] = data[0]  # DC component
        complex_data[-1] = data[1]  # Nyquist component

        for i in range(1, half_points):
            complex_data[i] = complex(data[i * 2], data[i * 2 + 1])

        # Inverse FFT
        return np.fft.irfft(complex_data, n=num_points) * num_points


def cart_to_polar(spectrum, half_length_fft):
    """Convert complex spectrum to polar form (magnitude/phase pairs)"""
    polar_spectrum = np.zeros((half_length_fft + 1) * 2)

    # Handle DC component (real only)
    polar_spectrum[0] = abs(spectrum[0])  # DC magnitude
    polar_spectrum[1] = 0.0  # DC phase (always 0)

    # Handle Nyquist component (real only)
    polar_spectrum[half_length_fft * 2] = abs(spectrum[1])  # Nyquist magnitude
    polar_spectrum[half_length_fft * 2 + 1] = 0.0  # Nyquist phase (always 0)

    # Convert remaining complex values to magnitude/phase pairs
    for band in range(1, half_length_fft):
        real_idx = band * 2
        imag_idx = real_idx + 1

        real_part = spectrum[real_idx]
        imag_part = spectrum[imag_idx]

        amp_idx = band * 2
        phase_idx = amp_idx + 1

        # Compute magnitude and phase
        polar_spectrum[amp_idx] = math.sqrt(real_part * real_part + imag_part * imag_part)
        if polar_spectrum[amp_idx] == 0.0:
            polar_spectrum[phase_idx] = 0.0
        else:
            polar_spectrum[phase_idx] = -math.atan2(imag_part, real_part)

    return polar_spectrum


def polar_to_cart(polar_spectrum, half_length_fft):
    """Convert polar form (magnitude/phase pairs) back to complex spectrum"""
    spectrum = np.zeros(half_length_fft * 2)

    # Process DC component (band 0)
    spectrum[0] = polar_spectrum[0] * math.cos(polar_spectrum[1])

    # Process Nyquist component
    spectrum[1] = polar_spectrum[half_length_fft * 2] * math.cos(polar_spectrum[half_length_fft * 2 + 1])

    # Process remaining bands
    for band in range(1, half_length_fft):
        amp_idx = band * 2
        phase_idx = amp_idx + 1

        amplitude = polar_spectrum[amp_idx]
        phase = polar_spectrum[phase_idx]

        if amplitude == 0.0:
            real_value = 0.0
            imag_value = 0.0
        else:
            real_value = amplitude * math.cos(phase)
            imag_value = -amplitude * math.sin(phase)

        real_idx = band * 2
        imag_idx = real_idx + 1

        spectrum[real_idx] = real_value
        spectrum[imag_idx] = imag_value

    return spectrum


def phase_interpolate(polar_spectrum, last_phase_in, last_phase_out, pi_info):
    """Interpolate phases between analysis and synthesis frames"""
    # Phase per band is related to the time between frames
    phase_per_band = ((pi_info.decimation * TWO_PI) / pi_info.points)

    # Process each bin
    for band in range(pi_info.half_points + 1):
        amp_idx = band * 2
        phase_idx = amp_idx + 1

        # If no energy at this frequency, maintain the phase
        if polar_spectrum[amp_idx] == 0.0:
            phase_difference = 0.0
            polar_spectrum[phase_idx] = last_phase_out[band]
        else:
            # Phase locking is an option in the original code but not fully implemented
            # For simplicity, we're implementing the non-phase-locking version
            phase_difference = polar_spectrum[phase_idx] - last_phase_in[band]

            # Save current phase for next frame
            last_phase_in[band] = polar_spectrum[phase_idx]

            # Unwrap phase differences
            while phase_difference > PI:
                phase_difference -= TWO_PI
            while phase_difference < -PI:
                phase_difference += TWO_PI

            # Scale phase difference by time stretch factor
            phase_difference *= pi_info.scale_factor

            # Create new phase from interpolation/decimation ratio
            polar_spectrum[phase_idx] = last_phase_out[band] + phase_difference

            # Unwrap phase to stay within -π to π
            while polar_spectrum[phase_idx] > PI:
                polar_spectrum[phase_idx] -= TWO_PI
            while polar_spectrum[phase_idx] < -PI:
                polar_spectrum[phase_idx] += TWO_PI

            # Store new phase for next frame
            last_phase_out[band] = polar_spectrum[phase_idx]

    return polar_spectrum


def simple_spectral_gate(polar_spectrum, pi_info):
    """Apply spectral gating to reduce noise"""
    if pi_info.min_amplitude == 0.0 and pi_info.mask_ratio == 0.0:
        return polar_spectrum

    # Find maximum amplitude
    max_amplitude = 0.0
    for band in range(pi_info.half_points + 1):
        amp_idx = band * 2
        if polar_spectrum[amp_idx] > max_amplitude:
            max_amplitude = polar_spectrum[amp_idx]

    # Calculate mask amplitude
    mask_amplitude = pi_info.mask_ratio * max_amplitude

    # Apply gating
    for band in range(pi_info.half_points + 1):
        amp_idx = band * 2
        if polar_spectrum[amp_idx] < mask_amplitude or polar_spectrum[amp_idx] < pi_info.min_amplitude:
            polar_spectrum[amp_idx] = 0.0

    return polar_spectrum


def add_synth(polar_spectrum, output, last_amp, last_freq, last_phase, sine_index, sine_table, pi_info):
    """Oscillator bank resynthesizer for pitch shifting"""
    one_over_interp = 1.0 / pi_info.interpolation
    cycles_band = pi_info.scale_factor * 8192 / pi_info.points
    cycles_frame = pi_info.scale_factor * 8192 / (pi_info.decimation * TWO_PI)

    # Determine number of partials to synthesize
    if pi_info.scale_factor > 1.0:
        number_partials = int(pi_info.half_points / pi_info.scale_factor)
    else:
        number_partials = pi_info.half_points

    # Create a copy of the output buffer to work with
    output_buffer = np.copy(output)

    # Process each partial/band
    for band in range(number_partials):
        amp_idx = band * 2
        freq_idx = amp_idx + 1

        # Start where we left off
        address = sine_index[band]

        if polar_spectrum[amp_idx] == 0.0:
            # Just set frequency for next interpolation
            polar_spectrum[freq_idx] = band * cycles_band
        else:
            # Calculate phase difference and unwrap
            phase_difference = polar_spectrum[freq_idx] - last_phase[band]
            last_phase[band] = polar_spectrum[freq_idx]

            while phase_difference > PI:
                phase_difference -= TWO_PI
            while phase_difference < -PI:
                phase_difference += TWO_PI

            # Convert to instantaneous frequency
            polar_spectrum[freq_idx] = phase_difference * cycles_frame + band * cycles_band

            # Start with last amplitude
            amplitude = last_amp[band]

            # Increment per sample to get to new frequency
            amp_increment = (polar_spectrum[amp_idx] - amplitude) * one_over_interp

            # Start with last frequency
            frequency = last_freq[band]

            # Increment per sample to get to new frequency
            freq_increment = (polar_spectrum[freq_idx] - frequency) * one_over_interp

            # Fill the output with one sine component
            for sample in range(pi_info.interpolation):
                # Get the sine value from the table with linear interpolation
                table_idx = int(address)
                frac = address - table_idx
                table_idx = table_idx % 8192
                next_idx = (table_idx + 1) % 8192

                sine_val = sine_table[table_idx] * (1 - frac) + sine_table[next_idx] * frac
                output_buffer[sample] += amplitude * sine_val

                # Update phase
                address += frequency
                while address >= 8192:
                    address -= 8192
                while address < 0:
                    address += 8192

                # Update amplitude and frequency
                amplitude += amp_increment
                frequency += freq_increment

        # Save current values for next iteration
        last_freq[band] = polar_spectrum[freq_idx]
        last_amp[band] = polar_spectrum[amp_idx]
        sine_index[band] = address

    return output_buffer


def overlap_add(input_data, synthesis_window, output_buffer, current_time, points, window_size):
    """Overlap-add windowed data into output buffer"""
    # Normalize time pointer
    while current_time < 0:
        current_time += points
    current_time %= points

    # Perform overlap-add
    for i in range(window_size):
        output_buffer[i] += input_data[current_time] * synthesis_window[i]
        current_time = (current_time + 1) % points

    return output_buffer


def shift_out(output_buffer, interpolation, window_size):
    """Shift out interpolation samples from buffer"""
    # Extract output samples
    output_samples = output_buffer[:interpolation].copy()

    # Shift buffer left
    output_buffer[:window_size - interpolation] = output_buffer[interpolation:window_size]
    output_buffer[window_size - interpolation:] = 0.0

    return output_samples, output_buffer


def find_best_ratio(pi_info):
    """Find the best ratio of interpolation/decimation for a given scale factor"""
    if pi_info.time:
        if pi_info.scale_factor > 1.0:
            # For time stretching by more than 1.0
            max_interpolate = pi_info.window_size // 8
            percent_error = 2.0

            for interp in range(max_interpolate, 0, -1):
                decimation = int(interp / pi_info.scale_factor)
                if decimation == 0:
                    decimation = 1
                test_scale = interp / decimation

                # Calculate error percentage
                if test_scale > pi_info.scale_factor:
                    percent_error = test_scale / pi_info.scale_factor
                else:
                    percent_error = pi_info.scale_factor / test_scale

                if percent_error < 1.004:
                    pi_info.interpolation = interp
                    pi_info.decimation = decimation
                    break

                if interp == 1:
                    pi_info.interpolation = max_interpolate
                    pi_info.decimation = int(pi_info.interpolation / pi_info.scale_factor)
                    if pi_info.decimation == 0:
                        pi_info.decimation = 1
                    percent_error = 1.0
        else:
            # For time compression (scale factor < 1.0)
            max_decimate = pi_info.window_size // 8
            percent_error = 2.0

            for decimation in range(max_decimate, 0, -1):
                interp = int(decimation * pi_info.scale_factor)
                if interp == 0:
                    interp = 1
                test_scale = interp / decimation

                # Calculate error percentage
                if test_scale > pi_info.scale_factor:
                    percent_error = test_scale / pi_info.scale_factor
                else:
                    percent_error = pi_info.scale_factor / test_scale

                if percent_error < 1.004:
                    pi_info.decimation = decimation
                    pi_info.interpolation = interp
                    break

                if decimation == 1:
                    pi_info.decimation = max_decimate
                    pi_info.interpolation = int(pi_info.decimation * pi_info.scale_factor)
                    if pi_info.interpolation == 0:
                        pi_info.interpolation = 1
                    percent_error = 1.0
    else:
        # For pitch shifting, use equal decimation and interpolation
        max_interpolate = pi_info.window_size // 8
        pi_info.decimation = pi_info.interpolation = max_interpolate

    # Make sure we have safe values
    if pi_info.decimation <= 0:
        pi_info.decimation = 1
    if pi_info.interpolation <= 0:
        pi_info.interpolation = 1

    # Calculate the actual scale factor achieved
    new_scale_factor = pi_info.interpolation / pi_info.decimation
    return new_scale_factor


def process_phase_vocoder(input_file, output_file, scale_factor, mode, window_type=VONHANN, fft_size=1024, 
                          min_amplitude=0.0, mask_ratio=0.0, overlap=4, preserve_stereo=False):
    """
    Process audio file with phase vocoder
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        scale_factor: Scale factor for time-stretching or pitch-shifting
        mode: TIME_MODE for time-stretching, PITCH_MODE for pitch-shifting
        window_type: Type of window function to use
        fft_size: Size of FFT (powers of 2 work best)
        min_amplitude: Minimum amplitude threshold for spectral gating
        mask_ratio: Ratio of max amplitude for masking
        overlap: Window overlap factor (window_size = fft_size * overlap)
        preserve_stereo: If True, process stereo files as stereo
    """
    # Read input file
    try:
        samples, sample_rate = sf.read(input_file, dtype='float32')
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
    
    # Determine number of channels
    if len(samples.shape) > 1:
        n_channels = samples.shape[1]
    else:
        n_channels = 1
        # Reshape to 2D array with 1 channel for consistent handling
        samples = samples.reshape(-1, 1)
    
    # Initialize parameters
    pi_info = PvocInfo()
    pi_info.points = fft_size
    pi_info.half_points = fft_size // 2
    pi_info.overlap = overlap
    pi_info.window_size = fft_size * overlap
    pi_info.window_type = window_type
    pi_info.time = mode
    pi_info.scale_factor = scale_factor
    pi_info.min_amplitude = min_amplitude
    pi_info.mask_ratio = mask_ratio
    
    # Find optimal decimation/interpolation values
    pi_info.scale_factor = find_best_ratio(pi_info)
    
    print(f"Using decimation={pi_info.decimation}, interpolation={pi_info.interpolation}")
    print(f"Actual scale factor: {pi_info.scale_factor}")
    
    # Create windows
    analysis_window = get_window(pi_info.window_size, pi_info.window_type)
    synthesis_window = get_window(pi_info.window_size, pi_info.window_type)
    
    # Scale the windows
    scale_windows(analysis_window, synthesis_window, pi_info)
    
    # Output channels
    channels_to_process = min(n_channels, 2) if preserve_stereo else 1
    if n_channels > 1 and not preserve_stereo:
        print(f"Input has {n_channels} channels, but preserve_stereo is False. Processing first channel only.")
    
    # Prepare output channels
    output_channels = []
    
    # Process each channel
    for channel in range(channels_to_process):
        print(f"\nProcessing channel {channel+1}/{channels_to_process}...")
        
        # Initialize processing state
        in_pointer = -pi_info.window_size
        out_pointer = (in_pointer * pi_info.interpolation) // pi_info.decimation
        in_position = 0
        
        # Input and output buffers
        input_buffer = np.zeros(pi_info.window_size)
        output_buffer = np.zeros(pi_info.window_size)
        
        # Phase tracking arrays
        last_phase_in = np.zeros(pi_info.half_points + 1)
        last_phase_out = np.zeros(pi_info.half_points + 1)
        
        # For additive synthesis (pitch shifting)
        last_amp = np.zeros(pi_info.half_points + 1)
        last_freq = np.zeros(pi_info.half_points + 1)
        sine_index = np.zeros(pi_info.half_points + 1)
        sine_table = np.array([0.5 * math.cos(i * TWO_PI / 8192) for i in range(8192)])
        
        # Output sample array
        output_samples = []
        
        # Print processing information
        if pi_info.time:
            print(f"Changing length by factor: {pi_info.scale_factor}")
        else:
            print(f"Changing pitch by factor: {pi_info.scale_factor}")
        
        # Main processing loop
        block_count = 0
        
        while True:
            # Increment times
            in_pointer += pi_info.decimation
            out_pointer += pi_info.interpolation
            
            # Read and process input 
            result = shift_in(samples, in_position, pi_info.window_size, pi_info.decimation, channel)
            
            if result == -2:
                # End of file
                break
                
            input_buffer, num_read = result
            
            if num_read <= 0 and in_position > 0:
                # End of file
                break
                
            # Window and fold input into FFT buffer
            spectrum = window_fold(input_buffer, analysis_window, in_pointer, pi_info.points, pi_info.window_size)
            
            # FFT: time-domain to frequency-domain
            spectrum = real_fft(spectrum, pi_info.points, TIME2FREQ)
            
            # Convert to amplitude/phase representation
            polar_spectrum = cart_to_polar(spectrum, pi_info.half_points)
            
            # Apply spectral gate if needed
            polar_spectrum = simple_spectral_gate(polar_spectrum, pi_info)
            
            # Process based on mode (time-stretching or pitch-shifting)
            if pi_info.time:
                # Phase vocoder for time-stretching
                polar_spectrum = phase_interpolate(polar_spectrum, last_phase_in, last_phase_out, pi_info)
                
                # Convert back to complex spectrum
                spectrum = polar_to_cart(polar_spectrum, pi_info.half_points)
                
                # IFFT: frequency-domain to time-domain
                time_domain = real_fft(spectrum, pi_info.points, FREQ2TIME)
                
                # Overlap-add to output buffer
                output_buffer = overlap_add(time_domain, synthesis_window, output_buffer, out_pointer, pi_info.points, pi_info.window_size)
            else:
                # Additive synthesis for pitch-shifting
                output_buffer = add_synth(polar_spectrum, output_buffer, last_amp, last_freq, 
                                        last_phase_in, sine_index, sine_table, pi_info)
            
            # Shift out samples from output buffer
            output_samples_block, output_buffer = shift_out(
                output_buffer, 
                pi_info.interpolation,
                pi_info.window_size
            )
            
            # Add to output sample array
            output_samples.extend(output_samples_block)
            
            # Update position counters
            in_position += pi_info.decimation
            block_count += 1
            
            # Print progress
            if block_count % 10 == 0:
                percent_done = min(100, (in_position / len(samples)) * 100)
                print(f"Processing: {percent_done:.1f}% complete", end='\r')
        
        # Store the processed channel
        output_channels.append(np.array(output_samples, dtype=np.float32))
    
    print("\nWriting output file...")
    
    # Prepare output array
    if channels_to_process > 1:
        # Ensure both channels have the same length
        min_length = min(len(channel) for channel in output_channels)
        output_array = np.column_stack([channel[:min_length] for channel in output_channels])
    else:
        output_array = output_channels[0]
    
    # Normalize output if needed
    max_val = np.max(np.abs(output_array))
    if max_val > 0.0:  # Avoid division by zero
        if max_val > 1.0:
            output_array = output_array / max_val
            print(f"Normalized output (max value was {max_val})")
        
    # Write output file
    try:
        sf.write(output_file, output_array, sample_rate)
        print(f"Wrote {len(output_array)} samples to {output_file}")
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


def main():
    """Process command-line arguments and run the phase vocoder"""
    parser = argparse.ArgumentParser(description="Python implementation of SoundHack's Phase Vocoder")
    
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("output_file", help="Output audio file")
    parser.add_argument("-s", "--scale", type=float, default=1.0, 
                      help="Scale factor (>1.0 stretches time or lowers pitch, <1.0 compresses time or raises pitch)")
    parser.add_argument("-m", "--mode", choices=["time", "pitch"], default="time",
                      help="Processing mode: 'time' for time-stretching, 'pitch' for pitch-shifting")
    parser.add_argument("-w", "--window", type=int, choices=range(1, 8), default=2,
                      help="Window type: 1=Hamming, 2=Hann, 3=Kaiser, 4=Ramp, 5=Rectangle, 6=Sinc, 7=Triangle")
    parser.add_argument("-f", "--fft-size", type=int, default=1024,
                      help="FFT size (power of 2 recommended)")
    parser.add_argument("-o", "--overlap", type=int, default=4,
                      help="Window overlap factor (window_size = fft_size * overlap)")
    parser.add_argument("-a", "--min-amplitude", type=float, default=0.0,
                      help="Minimum amplitude threshold for spectral gating")
    parser.add_argument("-r", "--mask-ratio", type=float, default=0.0,
                      help="Ratio of max amplitude for masking in spectral gating")
    parser.add_argument("--stereo", action="store_true", default=False,
                      help="Preserve stereo channels if present")
    
    args = parser.parse_args()
    
    # Validate FFT size (should be power of 2)
    if not (args.fft_size & (args.fft_size - 1) == 0) or args.fft_size <= 0:
        print(f"Warning: FFT size {args.fft_size} is not a power of 2. This may cause issues.")
    
    # Set mode flag
    mode = TIME_MODE if args.mode == "time" else PITCH_MODE
    
    # Process with phase vocoder
    success = process_phase_vocoder(
        args.input_file,
        args.output_file,
        args.scale,
        mode,
        args.window,
        args.fft_size,
        args.min_amplitude,
        args.mask_ratio,
        args.overlap,
        args.stereo
    )
    
    if success:
        print("Processing completed successfully.")
    else:
        print("Processing failed.")
    
    return 0 if success else 1


if __name__ == "__main__":
    main()