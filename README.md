# GradESTC

**There seems to be some problem with anonymous github.**
**If you can't read the source code online, please download it and view it. We are trying to fix the problem.**

Code for paper “Communication-Efficient Federated Learning by Exploiting Spatio-Temporal Correlations”.

## Environment Preparation

```sh
pip install -r requirements.txt
```

## Easy Run

ALL classes of methods are inherited from `FedAvgServer` and `FedAvgClient`. 

### Step 1. Generate FL Dataset

Partition the MNIST according to Dir(0.1) for 100 clients

```shell
python generate_data.py -d mnist -cn 10 --iid --seed 42
python generate_data.py -d mnist -a 0.5 -cn 10 --seed 42
python generate_data.py -d cifar10 --iid -cn 10 --seed 42
python generate_data.py -d cifar10 -a 0.5 -cn 10 --seed 42
python generate_data.py -d cifar100 --iid -cn 10 --seed 42
python generate_data.py -d cifar100 -a 0.5 -cn 10 --seed 42
```

About methods of generating federated dastaset, go check [`data/README.md`](data/#readme) for full details. The generated data is mainly stored in ./data/dataset

### Step 2. Run Experiment

```sh
python main.py gradestc config/mnist_lenet5.yml
python main.py gradestc config/mnist_lenet5_noniid-0.5.yml
python main.py gradestc config/cifar10_resnet18.yml
python main.py gradestc config/cifar10_resnet18_noniid-0.5.yml
python main.py gradestc config/cifar100_alexnet.yml
python main.py gradestc config/cifar100_alexnet_noniid-0.5.yml
```
