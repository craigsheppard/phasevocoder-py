## Python implementation of SoundHack's Phase Vocoder

### positional arguments:
  input_file            Input audio file
  output_file           Output audio file

### options:
  -h, --help            show this help message and exit
  -s SCALE, --scale SCALE
                        Scale factor (>1.0 stretches time or lowers pitch,
                        <1.0 compresses time or raises pitch)
  -m {time,pitch}, --mode {time,pitch}
                        Processing mode: 'time' for time-stretching, 'pitch'
                        for pitch-shifting
  -w {1,2,3,4,5,6,7}, --window {1,2,3,4,5,6,7}
                        Window type: 1=Hamming, 2=Hann, 3=Kaiser, 4=Ramp,
                        5=Rectangle, 6=Sinc, 7=Triangle
  -f FFT_SIZE, --fft-size FFT_SIZE
                        FFT size (power of 2 recommended)
  -o OVERLAP, --overlap OVERLAP
                        Window overlap factor (window_size = fft_size *
                        overlap)
  -a MIN_AMPLITUDE, --min-amplitude MIN_AMPLITUDE
                        Minimum amplitude threshold for spectral gating
  -r MASK_RATIO, --mask-ratio MASK_RATIO
                        Ratio of max amplitude for masking in spectral gating
  --stereo              Preserve stereo channels if present

`python main.py amen.wav amen_stretch.wav -m time -s 8 -w 1 -f 4096 -o 8 --stereo `