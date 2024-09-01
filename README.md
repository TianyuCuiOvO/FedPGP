# Harmonizing Generalization and Personalization in Federated Prompt Learning [ICML2024]
The implementation of paper Harmonizing Generalization and Personalization in Federated Prompt Learning[ICML2024].
[[paper]](https://arxiv.org/abs/2405.09771)

## How to Run

You can run `federated_main.py` with some specified arguments.

## Data Preparation
Please follow the instructions at CoOP https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md to prepare the following datasets: Caltech101, OxfordPets, Flowers102, Food101, DTD.

For CIFAR10 and CIFAR100 datasets, please download and unzip data under `DATA/` file catalog. Or simply run experiments with CIFAR10/CIFAR100 dataset, the program will download data automatically.

For DomainNet and office-caltech10 datasets, please follow the instructions of Dataset described [here](https://github.com/med-air/FedBN/blob/master/README.md). 

### Training

`--root` takes as input a path to dataset.

`--config-file` means which config file to use.

You can select variables like shots, users by changing `cfg` or you can change every arguments you like in scripts.

### Running example
`bash scripts/plt_few_shot.sh`
## Citation
If you find our work useful in your research, please consider citing:
```
@article{cui2024harmonizing,
  title={Harmonizing Generalization and Personalization in Federated Prompt Learning},
  author={Cui, Tianyu and Li, Hongxia and Wang, Jingya and Shi, Ye},
  journal={arXiv preprint arXiv:2405.09771},
  year={2024}
}
```


