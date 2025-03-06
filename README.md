# Predicting the critical temperature of a superconductor

This repository contains the code for estimation of the statistical model to predict the superconducting critical temperature based on the features extracted from the superconductor’s chemical formula [by Hamidieh et al. (2018)](http://arxiv.org/pdf/1803.10260) published in [UCIML repository](https://archive.ics.uci.edu/dataset/464/superconductivty+data).

## Install
You will need python version between 3.9 and 3.11 (inclusive).
1. Create venv and install dependencies: 
    ```shell
    poetry install
    ```

## Prepare data
1. ```shell
    chmod +x ./scripts/data/*
    ```
2. Download data:
    ```shell
    ./scripts/data/download.sh ./data
    ```
2. Process data:
    ```shell
    ./scripts/data/process.sh ./data
    ```

## Train
The configuration is managed by hydra framework. The original CLI produced by hydra is available in `superconduct_tc_reg.train` module.

The default configuration is stored under `./configs`.

1. Activate venv
    ```
    poetry shell
    ```
2. Launch train of gbdt model overriding directories: 
    ```
    python -m superconduct_tc_reg.train pipeline=xgb dir.base=$(pwd)
    ```

## References

- Kam Hamidieh, . "A data-driven statistical model for predicting the critical temperature of a superconductor".Computational Materials Science (2018).

- Yu, Jiahao, Yongman, Zhao, Rongshun, Pan, Xue, Zhou, and Zikai, Wei. "Prediction of the Critical Temperature of Superconductors Based on Two-Layer Feature Selection and the Optuna-Stacking Ensemble Learning Model".ACS Omega 8, no.3 (2023): 3078-3090.