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



\## Results



After training the three models on the generated signal dataset, the final MSE losses were:



| Model | Final MSE Loss |

|---|---:|

| MLP | 0.009813 |

| RNN | 0.005442 |

| LSTM | 0.006107 |



In this run, the RNN achieved the lowest MSE loss. This may happen because the context window is short, only 10 samples, so a simple recurrent memory is enough for this task. LSTM is more powerful for longer sequences, but for a short window it does not always outperform RNN.









\## How to run

```bash

python -m src.main

```



To train the models and compare the losses:



```bash

python -m src.train

```



\## How to test



```bash

python -m pytest

```



Current test result:



```text

8 passed

```



\## Design Choices

Some assignment details were not explicitly defined, so the following choices were made:



\- Four frequencies were selected: 1 Hz, 2 Hz, 5 Hz and 10 Hz.

\- The context window size is 10 samples, according to the assignment.

\- Noise is generated randomly using a sigma value.

\- The task is interpreted as reconstructing clean signal samples from noisy signal samples.

\- MSE is used as the loss function, according to the assignment.

\- PyTorch is used to build and train the neural network models.



\## Project Structure



text

uoh-rn01-ex01/

&#x20; README.md

&#x20; .gitignore

&#x20; docs/

&#x20;   PRD.md

&#x20;   PLAN.md

&#x20;   TODO.md

&#x20; src/

&#x20;   main.py

&#x20;   models.py

&#x20;   train.py

&#x20; tests/

&#x20;   test\_basic.py

&#x20;   test\_models.py





\## GitHub Link

https://github.com/layanawawde/uoh-rn01-ex01

