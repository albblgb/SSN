# Steganography of Steganographic Networks
This repo is the official code for

* [**Steganography of Steganographic Networks.**](https://ojs.aaai.org/index.php/AAAI/article/view/25647) 

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

## Citation
If you find our paper or code useful for your research, please cite:
```
@article{Li_Li_Li_Zhang_Qian_2023,
    author={Li, Guobiao and Li, Sheng and Li, Meiling and Zhang, Xinpeng and Qian, Zhenxing},
    title={Steganography of Steganographic Networks},
    volume={37},
    url={https://ojs.aaai.org/index.php/AAAI/article/view/25647},
    DOI={10.1609/aaai.v37i4.25647},
    number={4},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2023},
    month={Jun.},
    pages={5178-5186}
}
```
