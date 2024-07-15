import os
import numpy as np
import scipy
import argparse


def normalise_channel(signal):
    signal = np.array(signal, dtype=np.float32)
    signal /= np.max(np.abs(signal))
    return signal


def mix_with_snr(signal, noise, target_snr):
    # Calculate signal power and noise power
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = np.sum(noise ** 2) / len(noise)
    
    # Calc SNR with no scaling applied to audio files 
    current_snr = 10 * np.log10(signal_power / noise_power)
    
    # Calculate scaling factor for noise to achieve target SNR after summation
    scaling_factor = np.sqrt(10 ** ((current_snr - target_snr) / 10))
    
    # Mix noise to the signal
    mixed_signal = signal + scaling_factor * noise
    
    return normalise_channel(mixed_signal)


def add_noise(signal, snr, noise_filepath):
    # Load babble
    print(f"Retrieving noise from: {noise_filepath} Targetting SNR: {snr}")
    print(scipy.io.wavfile.read(noise_filepath))
    sampling_freq, noise = scipy.io.wavfile.read(noise_filepath)
    _, noise_ch = noise.shape
    signal_length, signal_ch = signal.shape
    print(f"Noise parsed - {noise_ch}ch @ {sampling_freq}Hz")
    if noise_ch != signal_ch:
        raise Exception(f"Noise and signal dimensions must agree. {noise_ch} vs {signal_ch}")

    out = np.zeros(signal.shape)
    for ch in range(noise_ch):
        normalised_noise = normalise_channel(noise[:, ch])
        out[:, ch] = mix_with_snr(signal[:, ch], normalised_noise[:signal_length], snr) 

    # Mix at the requested signal-to-noise ratio.
    return out


def main(impulse_response_mat_path, input_filepath, output_path, snr, noise_filepath):
    # Load IR
    ir = scipy.io.loadmat(impulse_response_mat_path)["response"]
    irLen, inputChannels = ir.shape
    
    # Load clean speech
    sampling_freq, dry_signal_raw = scipy.io.wavfile.read(input_filepath)
    print(f"Input sampling freq: {sampling_freq}")
    dry_signal = normalise_channel(dry_signal_raw)
    inputLen = dry_signal.shape[0]
    outputLen = irLen + inputLen - 1

    # Convolve with IR
    print("Convolving...")
    resultant_channels = np.zeros((outputLen, inputChannels), dtype=float)  # Initialize result array
    for col in range(inputChannels):
        print(f"Convolving input channel {col+1}/{inputChannels}")
        conv_result = np.convolve(ir[:, col], dry_signal, mode='full')
        resultant_channels[:, col] = conv_result
    print(resultant_channels)
    print(resultant_channels.shape)

    # import pickle
    # TEMP_CACHE_FILE = "temp_cache.pickle"
    # with open(TEMP_CACHE_FILE, 'wb') as pickle_file:
    #     pickle.dump(resultant_channels, pickle_file)

    # with open(TEMP_CACHE_FILE, "rb") as pickle_file:
    #     resultant_channels = pickle.load(pickle_file)

    # With noise
    print("Adding noise...")
    noise_mixed = None
    if snr is not None:
        noise_mixed = add_noise(resultant_channels, snr, noise_filepath)
        print(noise_mixed)
    
    # Contrast stretch back up to 16-bit
    output = np.array((noise_mixed if noise_mixed is not None else resultant_channels) * 32767, dtype=np.int16)

    output_path = output_path or f"{os.path.splitext(os.path.basename(input_filepath))[0]}_snr_{snr if snr is not None else "∞"}db.wav"
    scipy.io.wavfile.write(output_path, sampling_freq, output)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("impulse_response_mat")
    parser.add_argument("input_signal")
    parser.add_argument('-o', '--output')
    parser.add_argument('-s', '--snr', default=None, type=int)
    parser.add_argument('--noise', default="./resources/babble.wav")
    args = parser.parse_args()

    main(args.impulse_response_mat, args.input_signal, args.output, args.snr, args.noise)
