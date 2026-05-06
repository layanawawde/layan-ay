\# Assignment 01 - Signal Reconstruction



\## Group Code

uoh-rn01



\## What is this project?

This project generates simple sine and cosine signals, adds random noise to them, and trains three neural network models to reconstruct the clean signal.



The three models are:

1\. Fully Connected MLP

2\. RNN

3\. LSTM



\## Dataset

The dataset contains signal windows of 10 samples.



Each example contains:

\- a frequency encoded as one-hot vector

\- a noise level called sigma

\- 10 noisy samples

\- 10 clean samples



\## Frequencies

We use four known frequencies:

\- 1 Hz

\- 2 Hz

\- 5 Hz

\- 10 Hz



\## Loss Function

We use Mean Squared Error, MSE, because the model predicts numbers and we compare them to the clean signal.



\## How to run

```bash

python -m src.main

