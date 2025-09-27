#!/usr/bin/env python3
"""
shallow_cnn_rf_detection.py

This script simulates RF IQ data, converts signals to spectrograms,
builds a shallow CNN using TensorFlow/Keras, and includes a sliding-window
detector function. It is intended as an educational starting point.

Note: This script requires TensorFlow (or tensorflow-cpu) and common packages.
Install with:
pip install numpy scipy matplotlib scikit-learn tensorflow python-pptx python-docx

Usage:
  python3 shallow_cnn_rf_detection.py --train
  python3 shallow_cnn_rf_detection.py --detect --model model.h5
"""
import os
import argparse
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Try to import tensorflow; if not available, the script will only produce simulated data and spectrograms.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow not available. Model building/training sections will be skipped.")
    TF_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------
# Simulation utilities
# ---------------------
def simulate_tone(length, fs, freq, snr_db=30):
    t = np.arange(length) / fs
    tone = np.exp(2j * np.pi * freq * t)  # complex exponential (IQ)
    # Add AWGN
    sig_power = 1.0
    snr = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(t)) + 1j*np.random.randn(len(t)))
    return tone + noise

def simulate_jammer(length, fs, kind='broadband', snr_db=0):
    t = np.arange(length) / fs
    if kind == 'broadband':
        # wideband noise
        noise = (np.random.randn(len(t)) + 1j*np.random.randn(len(t)))
        # scale to desired SNR
        sig_power = 1.0
        snr = 10 ** (snr_db / 10.0)
        noise = noise / np.std(noise) * np.sqrt(sig_power / snr)
        return noise
    elif kind == 'tone':
        # strong tone jammer at random freq
        freq = np.random.uniform(-fs/2, fs/2)
        tone = 5.0 * np.exp(2j * np.pi * freq * t)
        return tone
    elif kind == 'pulsed':
        # pulsed high-power noise
        noise = np.zeros(len(t), dtype=complex)
        for _ in range(3):
            start = np.random.randint(0, len(t)//2)
            width = np.random.randint(len(t)//20, len(t)//5)
            noise[start:start+width] += (np.random.randn(width) + 1j*np.random.randn(width)) * 5.0
        return noise
    else:
        return (np.random.randn(len(t)) + 1j*np.random.randn(len(t)))

def iq_to_spectrogram(iq, fs, nfft=256, noverlap=192, nperseg=256):
    # Compute spectrogram of real and imaginary parts combined as magnitude
    f, t, Sxx = signal.spectrogram(iq, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Sxx = np.abs(Sxx)
    # Convert to log scale
    Sxx_log = 10 * np.log10(Sxx + 1e-12)
    # Normalize to 0-1
    Sxx_log = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-12)
    return Sxx_log

# ---------------------
# Dataset generation
# ---------------------
def generate_dataset(n_samples=200, length=4096, fs=1.0):
    X = []
    y = []
    for i in range(n_samples):
        if np.random.rand() < 0.5:
            # normal
            freq = np.random.uniform(-0.2*fs, 0.2*fs)
            snr_db = np.random.uniform(10, 40)
            sig = simulate_tone(length, fs, freq, snr_db=snr_db)
            label = 0
        else:
            # suspicious: pick jammer type
            kind = np.random.choice(['broadband', 'tone', 'pulsed'])
            sig = simulate_jammer(length, fs, kind=kind, snr_db=np.random.uniform(-5,10))
            label = 1
        S = iq_to_spectrogram(sig, fs=fs, nfft=256, noverlap=192, nperseg=256)
        # Resize/crop to fixed size (e.g., 64x64)
        S_resized = resize_spectrogram(S, (64,64))
        X.append(S_resized)
        y.append(label)
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y

def resize_spectrogram(S, target_shape=(64,64)):
    # Simple resizing by interpolation using numpy
    from scipy.ndimage import zoom
    zoom_f = (target_shape[0]/S.shape[0], target_shape[1]/S.shape[1])
    S2 = zoom(S, zoom_f, order=1)
    return S2

# ---------------------
# Model building
# ---------------------
def build_shallow_cnn(input_shape=(64,64,1), n_classes=2):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available.")
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------
# Sliding-window detector (pseudo real-time)
# ---------------------
def sliding_window_detector(iq_stream_generator, model, fs=1.0, window_length=4096, hop=2048, threshold=0.5, consecutive=3):
    """
    iq_stream_generator: yields complex IQ arrays (continuous stream).
    model: trained Keras model that returns probability for class 1 (suspicious).
    window_length: samples per window
    hop: step between windows
    threshold: probability threshold to mark suspicious
    consecutive: number of consecutive suspicious windows before alert
    """
    consec_count = 0
    buffer = np.array([], dtype=complex)
    for chunk in iq_stream_generator():
        buffer = np.concatenate([buffer, chunk])
        while len(buffer) >= window_length:
            window = buffer[:window_length]
            buffer = buffer[hop:]  # slide by hop
            S = iq_to_spectrogram(window, fs=fs, nfft=256, noverlap=192, nperseg=256)
            S2 = resize_spectrogram(S, (64,64))[np.newaxis,...,np.newaxis]
            prob = model.predict(S2)[0,1]
            print(f"Window prob suspicious: {prob:.3f}")
            if prob >= threshold:
                consec_count += 1
            else:
                consec_count = 0
            if consec_count >= consecutive:
                print("ALERT: Suspicious RF activity detected!")
                consec_count = 0

# ---------------------
# Main CLI
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Generate data and train model (small demo).')
    parser.add_argument('--detect', action='store_true', help='Run sliding-window detector on synthetic stream.')
    parser.add_argument('--model', type=str, default='model.h5', help='Path to save/load model.')
    args = parser.parse_args()

    if args.train:
        print("Generating dataset...")
        X, y = generate_dataset(n_samples=300, length=4096, fs=1.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if TF_AVAILABLE:
            model = build_shallow_cnn(input_shape=X_train.shape[1:], n_classes=2)
            print(model.summary())
            model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.1)
            model.save(args.model)
            print("Model saved to", args.model)
            # Evaluate
            preds = model.predict(X_test).argmax(axis=1)
            print(classification_report(y_test, preds))
            print("Confusion matrix:")
            print(confusion_matrix(y_test, preds))
        else:
            print("TensorFlow not available. Skipping training.")
            # Save simulated spectrogram examples to disk for inspection
            os.makedirs('spectrogram_examples', exist_ok=True)
            for i in range(min(10, X.shape[0])):
                plt.imsave(f'spectrogram_examples/spect_{i}.png', X[i,:,:,0], cmap='viridis')
            print("Saved example spectrograms to ./spectrogram_examples")

    if args.detect:
        if not TF_AVAILABLE:
            print("TensorFlow not available. Cannot run detector.")
            return
        if not os.path.exists(args.model):
            print("Model not found:", args.model)
            return
        model = tf.keras.models.load_model(args.model)
        # Create a simple generator that yields chunks containing a jammer after some time
        def gen():
            # first normal chunks
            for _ in range(3):
                yield simulate_tone(2048, fs=1.0, freq=np.random.uniform(-0.1,0.1), snr_db=30)
            # then jammed chunks
            for _ in range(10):
                yield simulate_jammer(2048, fs=1.0, kind='pulsed', snr_db=0)
            # normal again
            for _ in range(3):
                yield simulate_tone(2048, fs=1.0, freq=np.random.uniform(-0.1,0.1), snr_db=30)
        sliding_window_detector(gen, model, fs=1.0, window_length=4096, hop=2048, threshold=0.6, consecutive=2)

if __name__ == '__main__':
    main()
