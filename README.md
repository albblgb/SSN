# SSN
Code for "Steganography of Steganographic Networks"
    
## Dependencies and Installation
- Python 3.8.13, PyTorch = 1.11.0
- Run the following commands in your terminal:

  `conda env create -f env.yml`

  `conda activate pyt_env`


## Get Started
#### Sender: training and secret model and disgusing it to stego-model

- Run `python train_secret_model.py`,

- Run `python secret_model_disguising.py`,

#### Receiver extracting and testing the secret model form the stego-model

- Run `python secret_model_disguising.py`,


1. The secret and stego-models will be saved in 'checkpoint/'
2. The results will be saved in 'result/'



