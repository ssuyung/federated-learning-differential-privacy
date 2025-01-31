# Federated Learning with Differential Privacy: Decreasing Noise Mechanism

This is an implementation of **Federated Learning (FL)** with **Differential Privacy (DP)**. The FL algorithm is FedAvg, based on the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). Each client trains local model by DP-SGD [2] to perturb model parameters. The noise multiplier is determined by [3-5] (see rdp_analysis.py). 

## Introduction & Results
(Details in LSML_Final_Report.pdf)
Federated learning, combined with differential privacy, offers a robust approach to safeguarding sensitive data during distributed model training. In this study, we propose a decreasing noise mechanism to optimize the balance between privacy and model accuracy. By dynamically adjusting the gradient clipping size through a noise control parameter, noise_gamma, we explore the trade-offs between conver- gence rate and accuracy under fixed privacy budgets (ε, δ). Experiments conducted using Gaussian and Laplace noise on the MNIST dataset demonstrate that de- creasing noise improves model accuracy compared to fixed noise scales. Notably, Gaussian noise showed a slight accuracy enhancement for specific noise_gamma values, while Laplace noise yielded a modest 2–4% improvement. Our findings underline the importance of parameter tuning in achieving privacy-utility trade-offs, paving the way for future explorations into more complex datasets and models.

![laplace_decreasing_noise](https://github.com/user-attachments/assets/b2d2403e-ef0a-465c-a069-4acd537c4ef8)  
Laplace with decreasing noise, ε = 0.1  
![guassion_decresing _noise](https://github.com/user-attachments/assets/ed192cca-8e26-4e06-ae61-f37e06ecafcb)  
Gaussian with decreasing noise, ε = 0.5  


## Requirements
- torch, torchvision
- numpy
- scipy

## Files
> FLModel.py: definition of the FL client and FL server class

> MLModel.py: CNN model for MNIST datasets

> rdp_analysis.py: RDP for subsampled Gaussian [3], convert RDP to DP by Ref. [4, 5] (tighter privacy analysis than [2]).

> utils.py: sample MNIST in a non-i.i.d. manner

## Usage
Run test_cnn.ipynb

### FL model parameters
```python
# code segment in test_cnn.ipynb
lr = 0.1
fl_param = {
    'output_size': 10,          # number of units in output layer
    'client_num': client_num,   # number of clients
    'model': MNIST_CNN,  # model
    'data': d,           # dataset
    'lr': lr,            # learning rate
    'E': 500,            # number of local iterations
    'eps': 4.0,          # privacy budget
    'delta': 1e-5,       # approximate differential privacy: (epsilon, delta)-DP
    'q': 0.01,           # sampling rate
    'clip': 0.2,         # clipping norm
    'tot_T': 10,         # number of aggregation times (communication rounds)
    'epsilon': 1.0,
}
```


## References
[1] McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In *AISTATS*, 2017.

[2] Abadi, Martin, et al. Deep learning with differential privacy. In *CCS*. 2016.

[3] Mironov, Ilya, Kunal Talwar, and Li Zhang. R\'enyi differential privacy of the sampled gaussian mechanism. arXiv preprint 2019.

[4] Canonne, Clément L., Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential privacy. In *NeurIPS*, 2020.

[5] Asoodeh, S., Liao, J., Calmon, F.P., Kosut, O. and Sankar, L., A better bound gives a hundred rounds: Enhanced privacy guarantees via f-divergences. In *ISIT*, 2020.
