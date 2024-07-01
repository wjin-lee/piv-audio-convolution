import os
import numpy as np
import scipy
import argparse


def normalise_channels(signal):
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
    
    return normalise_channels(mixed_signal)


def add_noise(signal, snr, noise_filepath):
    # Load babble
    sampling_freq, babble_raw = scipy.io.wavfile.read(noise_filepath)
    print(f"SNR: {snr}")
    print(f"Noise sampling freq {sampling_freq}")
    babble = normalise_channels(np.mean(babble_raw, axis=1))
    signalLen = signal.shape[0]

    # Mix at the requested signal-to-noise ratio.
    return mix_with_snr(signal, babble[:signalLen], snr)


def main(impulse_response_mat_path, input_filepath, output_path, snr, noise_filepath):
    # Load IR
    ir = scipy.io.loadmat(impulse_response_mat_path)["response"]
    irLen, inputChannels = ir.shape
    
    # Load clean speech
    sampling_freq, dry_signal_raw = scipy.io.wavfile.read(input_filepath)
    dry_signal = normalise_channels(np.mean(dry_signal_raw, axis=1))
    inputLen = dry_signal.shape[0]

    # With noise
    if snr is not None:
        dry_signal = add_noise(dry_signal, snr, noise_filepath)


    # Convolve with IR
    print("Convolving...")
    resultant_channels = np.zeros(((max(irLen, inputLen)), ir.shape[1]), dtype=float)  # Initialize result array
    for col in range(inputChannels):
        print(f"Convolving input channel {col+1}/{inputChannels}")
        resultant_channels[:, col] = np.convolve(ir[:, col], dry_signal, mode='same')

    output = np.array(resultant_channels * 32767, dtype=np.int16)
    output_path = output_path or f"{os.path.splitext(os.path.basename(input_filepath))[0]}_snr_{snr}db.wav"
    scipy.io.wavfile.write(output_path, sampling_freq, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("impulse_response_mat")
    parser.add_argument("input_signal")
    parser.add_argument('-o', '--output')
    parser.add_argument('-s', '--snr', default=0, type=int)
    parser.add_argument('--noise', default="./resources/babble.wav")
    args = parser.parse_args()

    main(args.impulse_response_mat, args.input_signal, args.output, args.snr, args.noise)
