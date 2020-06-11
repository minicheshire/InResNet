# Interpolation between Residual and Non-Residual Networks

Zonghan Yang, Yang Liu, Chenglong Bao, and Zuoqiang Shi. https://arxiv.org/abs/2006.05749

## Accuracy

First of all, make directory according to the set random seed (i.e., RANDOM_SEED=0):

```
mkdir ./110layer/
mkdir ./110layer/seed0/
mkdir ./110layer/seed0/result/
```

Then train an In-ResNet or \lambda-In-ResNet (change the commented lines in train.py and InResNet.py accordingly). The accuracy over test set will be displayed at the end of training. 

```
python train.py [RANDOM_SEED] [SAVED_DIRECTORY] [MODEL_NAME] [LOG_FILE] [GPU_NO]
```

For example:
```
	[RANDOM_SEED] = "0"
	[SAVED_DIRECTORY] = "./110layer/seed0"
	[MODEL_NAME] = "In-ResNet-110"
	[LOG_FILE] = "./110layer/stats.txt"
	[GPU_NO] = "0"
```

## Robustness Against Stochastic Noise

Test the trained model on the stochastic noise groups in CIFAR-C:

```
python test_noise.py [MODEL_PATH] [GPU_NO] [LOG_FILE]
```

For example:
```
	[MODEL_PATH] = "./110layer/seed0/result/test-test1.mdlpkl"
	[GPU_NO] = "0"
	[LOG_FILE] = "./110layer/noise_stats.txt"
```

## Robustness Against Adversarial Attacks

Test the trained model against FGSM/IFGSM/PGD attack:

```
python test_{fgsm/ifgsm/pgd}.py [MODEL_PATH] [GPU_NO] [LOG_FILE] [RADIUS]
```

For example:
```
	[MODEL_PATH] = "./110layer/seed0/result/test-test1.mdlpkl"
	[GPU_NO] = "0"
	[LOG_FILE] = "./110layer/fgsm_stats.txt" # or "./110layer/ifgsm_stats.txt", "./110layer/pgd_stats.txt" 
	[RADIUS] = "8" # This means the attack radius $\epsilon = 8/255$
```

## Dependencies

```
PyTorch >= 1.2.0
```