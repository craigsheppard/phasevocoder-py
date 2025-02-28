#!/usr/bin/env python3
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import argparse


def sinc_window(window_size, alpha=3.0):
    """
    Create a sinc window.

    Parameters:
    -----------
    window_size : int
        Size of the window
    alpha : float
        Parameter controlling the width of the main lobe

    Returns:
    --------
    window : numpy array
        Sinc window values
    """
    # Create window indices centered at zero
    n = np.arange(window_size) - (window_size - 1) / 2

    # Avoid division by zero at center
    epsilon = 1e-10
    x = alpha * n / (window_size / 2)
    window = np.where(np.abs(x) < epsilon, 1.0, np.sin(np.pi * x) / (np.pi * x))

    # Apply Blackman window to reduce side lobes
    window = window * signal.windows.blackman(window_size)

    return window


def get_window(window_type, window_size, alpha=3.0):
    """
    Create a window of the specified type and size.

    Parameters:
    -----------
    window_type : str
        Type of window ('hann', 'hamming', 'blackman', 'rectangular', 'triangular', 'bartlett', 'kaiser', 'gaussian', 'sinc')
    window_size : int
        Size of the window
    alpha : float
        Parameter used for sinc window (default=3.0)

    Returns:
    --------
    window : numpy array
        Window function values
    """
    if window_type.lower() == 'hann':
        return signal.windows.hann(window_size)
    elif window_type.lower() == 'hamming':
        return signal.windows.hamming(window_size)
    elif window_type.lower() == 'blackman':
        return signal.windows.blackman(window_size)
    elif window_type.lower() == 'rectangular':
        return np.ones(window_size)
    elif window_type.lower() == 'triangular' or window_type.lower() == 'bartlett':
        return signal.windows.bartlett(window_size)
    elif window_type.lower() == 'kaiser':
        return signal.windows.kaiser(window_size, beta=8.6)
    elif window_type.lower() == 'gaussian':
        return signal.windows.gaussian(window_size, std=window_size / 6.0)
    elif window_type.lower() == 'sinc':
        return sinc_window(window_size, alpha)
    else:
        print(f"Unknown window type '{window_type}', defaulting to Hann window")
        return signal.windows.hann(window_size)


def phase_vocoder(x, stretch_factor, fft_size=2048, hop_size=None, overlap=0.75, window_type='hann', window_alpha=3.0):
    """
    Phase vocoder algorithm for time stretching without pitch change.

    Parameters:
    -----------
    x : numpy array
        Input audio signal
    stretch_factor : float
        Time stretching factor (>1 for slower, <1 for faster)
    fft_size : int
        Size of FFT window
    hop_size : int or None
        Hop size for analysis (if None, calculated from overlap)
    overlap : float
        Overlap factor (0.0 to 0.99, default 0.75)
    window_type : str
        Type of window to use (default 'hann')
    window_alpha : float
        Alpha parameter for sinc window

    Returns:
    --------
    y : numpy array
        Time-stretched output signal
    """
    # Calculate hop size from overlap if not provided
    if hop_size is None:
        hop_size = int(fft_size * (1 - overlap))

    # Ensure valid hop size
    hop_size = max(16, min(hop_size, fft_size - 1))

    # Calculate synthesis hop size with minimum to prevent extremely small hops
    synthesis_hop = max(16, int(hop_size / stretch_factor))

    # Get the specified window
    window = get_window(window_type, fft_size, window_alpha)

    # Normalize the window for energy preservation
    window = window / np.sqrt(np.sum(window ** 2) / hop_size) * np.sqrt(fft_size)

    # Number of frames for input and output
    num_frames_input = max(1, int((len(x) - fft_size) / hop_size) + 1)
    num_frames_output = max(1, int(num_frames_input * stretch_factor))

    # Initialize output array with proper size
    output_length = int(num_frames_output * synthesis_hop + fft_size)
    y = np.zeros(output_length)

    # Phase accumulation and phase advance arrays
    phase_accumulator = np.zeros(fft_size // 2 + 1)
    analysis_phase = np.zeros(fft_size // 2 + 1)
    synthesis_phase = np.zeros(fft_size // 2 + 1)

    # Frequency bins
    omega = 2 * np.pi * np.arange(fft_size // 2 + 1) / fft_size

    # Expected phase advance per hop at each bin
    phase_advance = omega * hop_size

    # Main processing loop
    for frame_index in range(num_frames_output):
        # Calculate corresponding input frame position using fixed rate conversion
        analysis_position = frame_index / stretch_factor
        analysis_frame = int(analysis_position)

        # Ensure we don't go past the input signal
        if analysis_frame >= num_frames_input:
            break

        # Get the frame
        start_sample = min(len(x) - fft_size, analysis_frame * hop_size)
        start_sample = max(0, start_sample)  # Ensure we don't go negative
        frame = x[start_sample:start_sample + fft_size]

        # Pad if necessary (if we're at the end of the signal)
        if len(frame) < fft_size:
            frame = np.pad(frame, (0, fft_size - len(frame)))

        # Apply window
        windowed_frame = frame * window

        # FFT
        spectrum = np.fft.rfft(windowed_frame)

        # Separate magnitude and phase
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Phase processing for proper synthesis
        if frame_index == 0:
            # First frame - just store the current phase
            analysis_phase = phase
            synthesis_phase = phase
        else:
            # Calculate phase increment
            phase_difference = phase - analysis_phase
            analysis_phase = phase

            # Unwrap phase differences to handle phase wrapping
            wrapped_phase_diff = (phase_difference + np.pi) % (2 * np.pi) - np.pi

            # Calculate true frequency
            true_freq_offset = wrapped_phase_diff - phase_advance

            # Update phase accumulator with the true frequency
            phase_accumulator += phase_advance + true_freq_offset * stretch_factor

            # Update synthesis phase
            synthesis_phase = phase_accumulator

        # Create new spectrum with adjusted phase
        new_spectrum = magnitude * np.exp(1j * synthesis_phase)

        # IFFT
        new_frame = np.real(np.fft.irfft(new_spectrum))

        # Apply window again for overlap-add
        new_frame *= window

        # Overlap-add to output
        start_idx = frame_index * synthesis_hop
        end_idx = min(len(y), start_idx + fft_size)
        y[start_idx:end_idx] += new_frame[:end_idx - start_idx]

    # Normalize output - more robust normalization for extreme stretch factors
    if np.max(np.abs(y)) > 1e-10:  # Avoid division by zero
        y = y * 0.95 / np.max(np.abs(y))  # Normalize to 95% of max amplitude

        # Scale to match input amplitude
        input_rms = np.sqrt(np.mean(x ** 2))
        output_rms = np.sqrt(np.mean(y ** 2))
        if output_rms > 1e-10:  # Avoid division by zero
            y = y * input_rms / output_rms

    return y


def pitch_shift(x, pitch_factor, sample_rate, fft_size=2048, hop_size=None, overlap=0.75, window_type='hann',
                window_alpha=3.0):
    """
    Pitch shift using phase vocoder and resampling.

    Parameters:
    -----------
    x : numpy array
        Input audio signal
    pitch_factor : float
        Pitch shifting factor (2.0 = one octave up, 0.5 = one octave down)
    sample_rate : int
        Sample rate of the audio
    fft_size : int
        Size of FFT window
    hop_size : int or None
        Hop size for analysis (if None, calculated from overlap)
    overlap : float
        Overlap factor (0.0 to 0.99, default 0.75)
    window_type : str
        Type of window to use (default 'hann')
    window_alpha : float
        Alpha parameter for sinc window

    Returns:
    --------
    y : numpy array
        Pitch-shifted output signal
    """
    # Time stretch
    stretched = phase_vocoder(x, 1.0 / pitch_factor, fft_size, hop_size, overlap, window_type, window_alpha)

    # Resample to original length
    from scipy import interpolate

    original_length = len(x)
    stretched_length = len(stretched)

    # Original time points
    time_orig = np.arange(stretched_length)

    # Time points for interpolation
    time_new = np.linspace(0, stretched_length - 1, int(stretched_length * pitch_factor))

    # Create interpolation function
    interpolator = interpolate.interp1d(time_orig, stretched, axis=0,
                                        bounds_error=False, fill_value=0)

    # Resample
    resampled = interpolator(time_new)

    # Trim or pad to match original length
    if len(resampled) > original_length:
        resampled = resampled[:original_length]
    else:
        resampled = np.pad(resampled, (0, original_length - len(resampled)))

    return resampled


def main():
    parser = argparse.ArgumentParser(description='Phase Vocoder for time-stretching and pitch-shifting audio files')
    parser.add_argument('input_file', help='Input audio file (WAV format)')
    parser.add_argument('output_file', help='Output audio file (WAV format)')
    parser.add_argument('--stretch', type=float, default=1.0,
                        help='Time stretch factor (>1 for slower, <1 for faster)')
    parser.add_argument('--pitch', type=float, default=1.0,
                        help='Pitch shift factor (2.0 = one octave up, 0.5 = one octave down)')
    parser.add_argument('--fft-size', type=int, default=2048, help='FFT window size')
    parser.add_argument('--hop-size', type=int, default=None,
                        help='Hop size (if not provided, calculated from overlap)')
    parser.add_argument('--overlap', type=float, default=0.75,
                        help='Overlap factor (0.0-0.99, default 0.75)')
    parser.add_argument('--window', type=str, default='hann', choices=[
        'hann', 'hamming', 'blackman', 'rectangular', 'triangular',
        'bartlett', 'kaiser', 'gaussian', 'sinc'
    ], help='Window type')
    parser.add_argument('--window-alpha', type=float, default=3.0,
                        help='Alpha parameter for sinc window (default 3.0)')
    parser.add_argument('--only-stretch', action='store_true', help='Only perform time stretching')
    parser.add_argument('--only-pitch', action='store_true', help='Only perform pitch shifting')

    args = parser.parse_args()

    # Validate overlap
    if args.overlap < 0.0 or args.overlap >= 1.0:
        print(f"Warning: Invalid overlap {args.overlap}, setting to default 0.75")
        args.overlap = 0.75

    # Read input file
    sample_rate, data = wavfile.read(args.input_file)

    # Convert to float
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0

    # Handle stereo input
    is_stereo = len(data.shape) > 1 and data.shape[1] > 1

    if is_stereo:
        left = data[:, 0]
        right = data[:, 1]

        # Process channels separately
        if args.only_stretch:
            left_processed = phase_vocoder(left, args.stretch, args.fft_size,
                                           args.hop_size, args.overlap, args.window, args.window_alpha)
            right_processed = phase_vocoder(right, args.stretch, args.fft_size,
                                            args.hop_size, args.overlap, args.window, args.window_alpha)
        elif args.only_pitch:
            left_processed = pitch_shift(left, args.pitch, sample_rate, args.fft_size,
                                         args.hop_size, args.overlap, args.window, args.window_alpha)
            right_processed = pitch_shift(right, args.pitch, sample_rate, args.fft_size,
                                          args.hop_size, args.overlap, args.window, args.window_alpha)
        else:
            # Apply both transformations
            left_stretched = phase_vocoder(left, args.stretch, args.fft_size,
                                           args.hop_size, args.overlap, args.window, args.window_alpha)
            right_stretched = phase_vocoder(right, args.stretch, args.fft_size,
                                            args.hop_size, args.overlap, args.window, args.window_alpha)

            left_processed = pitch_shift(left_stretched, args.pitch, sample_rate, args.fft_size,
                                         args.hop_size, args.overlap, args.window, args.window_alpha)
            right_processed = pitch_shift(right_stretched, args.pitch, sample_rate, args.fft_size,
                                          args.hop_size, args.overlap, args.window, args.window_alpha)

        # Make sure both channels have the same length
        min_length = min(len(left_processed), len(right_processed))
        left_processed = left_processed[:min_length]
        right_processed = right_processed[:min_length]

        # Combine channels
        processed = np.column_stack((left_processed, right_processed))
    else:
        # Mono processing
        if args.only_stretch:
            processed = phase_vocoder(data, args.stretch, args.fft_size,
                                      args.hop_size, args.overlap, args.window, args.window_alpha)
        elif args.only_pitch:
            processed = pitch_shift(data, args.pitch, sample_rate, args.fft_size,
                                    args.hop_size, args.overlap, args.window, args.window_alpha)
        else:
            # Apply both transformations
            stretched = phase_vocoder(data, args.stretch, args.fft_size,
                                      args.hop_size, args.overlap, args.window, args.window_alpha)
            processed = pitch_shift(stretched, args.pitch, sample_rate, args.fft_size,
                                    args.hop_size, args.overlap, args.window, args.window_alpha)

    # Convert back to int16
    processed = np.clip(processed, -1.0, 1.0)
    processed = (processed * 32767).astype(np.int16)

    # Calculate actual hop size used
    hop_size_used = args.hop_size if args.hop_size is not None else int(args.fft_size * (1 - args.overlap))

    # Write output file
    wavfile.write(args.output_file, sample_rate, processed)

    print(f"Processing complete: {args.input_file} → {args.output_file}")
    print(
        f"Parameters: FFT size: {args.fft_size}, Hop size: {hop_size_used}, Overlap: {args.overlap:.2f}, Window: {args.window}")
    if args.window == 'sinc':
        print(f"Sinc window alpha: {args.window_alpha}")
    if args.only_stretch:
        print(f"Time stretch factor: {args.stretch}")
    elif args.only_pitch:
        print(f"Pitch shift factor: {args.pitch}")
    else:
        print(f"Time stretch factor: {args.stretch}, Pitch shift factor: {args.pitch}")


if __name__ == "__main__":
    main()