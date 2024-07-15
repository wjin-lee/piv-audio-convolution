This is a quick script to convolve a given audio signal with a given noise signal at a requested signal-to-noise ratio (SNR).

## Usage
> [!NOTE]  
> This project has only been tested on Python version 3.12. Your milage may vary when using other (especially lower) Python versions.

1. Install Dependencies
   ```sh
   cd ./piv-audio-convolution
   pip install -r requirements.txt
   ```
2. Run the command utility
   ```sh
   python ./main.py <IMPULSE_RESPONSE_MAT_FILEPATH> <INPUT_SIGNAL_FILEPATH> [--snr <SNR>] [--noise <NOISE_FILEPATH>] [--output <OUTPUT_FILEPATH>]
   ```

    `impulse_response_mat` is the `.mat` file generated by the ScanIR program encoding the room impulse response. 
    
    `input_signal` is the 'pure' input speech signal to convolve with the impulse response.

    `--snr` option allows you to specify the signal-to-noise ratio. Defaults to `∞ db` (no noise).

    `--noise` option allows you to specify the noise file used. Defaults to `./resources/babble.wav`

    `--output` option allows you to specify the output filepath.