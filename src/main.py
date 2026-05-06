import math
import random


FREQUENCIES = [1, 2, 5, 10]
WINDOW_SIZE = 10
SAMPLE_RATE = 100


def one_hot(index, size=4):
    vector = [0] * size
    vector[index] = 1
    return vector


def generate_clean_signal(frequency, samples=WINDOW_SIZE):
    signal = []

    for i in range(samples):
        t = i / SAMPLE_RATE
        sine_part = math.sin(2 * math.pi * frequency * t)
        cosine_part = 0.5 * math.cos(2 * math.pi * frequency * t)
        value = sine_part + cosine_part
        signal.append(value)

    return signal


def add_noise(signal, sigma=0.1):
    noisy_signal = []

    for value in signal:
        noise = random.uniform(-sigma, sigma)
        noisy_signal.append(value + noise)

    return noisy_signal


def create_example(frequency_index=0, sigma=0.1):
    frequency = FREQUENCIES[frequency_index]
    clean_samples = generate_clean_signal(frequency)
    noisy_samples = add_noise(clean_samples, sigma)

    return {
        "frequency_one_hot": one_hot(frequency_index),
        "sigma": sigma,
        "noisy_samples": noisy_samples,
        "clean_samples": clean_samples,
    }


def main():
    example = create_example(frequency_index=0, sigma=0.1)

    print("Frequency one-hot:")
    print(example["frequency_one_hot"])

    print("\nSigma:")
    print(example["sigma"])

    print("\nNoisy samples:")
    print(example["noisy_samples"])

    print("\nClean samples:")
    print(example["clean_samples"])


if __name__ == "__main__":
    main()