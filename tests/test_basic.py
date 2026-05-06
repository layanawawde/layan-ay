from src.main import (
    one_hot,
    generate_clean_signal,
    add_noise,
    create_example,
    WINDOW_SIZE,
)


def test_one_hot_length():
    vector = one_hot(0)
    assert len(vector) == 4


def test_one_hot_value():
    vector = one_hot(2)
    assert vector == [0, 0, 1, 0]


def test_clean_signal_length():
    signal = generate_clean_signal(1)
    assert len(signal) == WINDOW_SIZE


def test_noise_length():
    clean = generate_clean_signal(1)
    noisy = add_noise(clean, sigma=0.1)
    assert len(noisy) == len(clean)


def test_example_structure():
    example = create_example(0, sigma=0.1)

    assert len(example["frequency_one_hot"]) == 4
    assert len(example["noisy_samples"]) == WINDOW_SIZE
    assert len(example["clean_samples"]) == WINDOW_SIZE
    assert example["sigma"] == 0.1