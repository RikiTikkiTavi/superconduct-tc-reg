# Predicting the critical temperature of a superconductor

This repository contains the code to train and deploy ML model to predict the superconducting critical temperature based on the features extracted from the superconductor’s chemical formula [by Hamidieh et al. (2018)](http://arxiv.org/pdf/1803.10260) published in [UCIML repository](https://archive.ics.uci.edu/dataset/464/superconductivty+data).

## Results

The main focus of this contribution is flexible model training & deployment & experimentation pipeline for selected modelling approaches.

**Engineering:**

Utilized technologies: `python`, `MLflow`, `torch`, `XGBoost`, `ONNX`, `scikit-learn`, `pandas`, `numpy`.

- Flexible model training & experimentation pipeline for DNN, GBDT and RF.
- Experiment tracking using MLflow
- Model versioning & deployment using mlflow and ONNX.

TODO: Proper model evaluation (at least `mlflow.evaluate`)


**Data analysis:**

1. There are 4025 rows (roughly 20% of the dataset), which refer to same element, but with significantly different (>5K) critical temperature.
2. Many highly correlated features arrising due to physical nature of the features and how the dataset was originally built.
3. There are "outliers" - entries, where critical temperature of the material was measured under extreme conditions with respect to other measurements of the same material. 
4. 99% of the variance is explained by the first 31 principal components (PCA). However, modelling reveals, that the rest PCs contain important information.
5. Cluster analysis did not reveal any cluster structures.

**Modelling:**

Some experiment results available at: https://dagshub.com/RikiTikkiTavi/superconduct-tc-reg/experiments (due to the latency of dagshub's mlflow service current number of experiments is limited.)

DNN and GBDT (Gradient-Boosted Decision Trees) models were compared under varying data processing steps and model architecture heuristics.
GBDT performs significantly better then DNN, while ensuring lower variance: val_RMSE 
(for now: bare eye estimate of sample mean and 98% CI from the box plot) $` \approx 9.6 \pm 0.2 `$ vs $` 13.8 \pm 0.5`$.
DNN requires slightly larger model size then GBDT and resource-intensive hyperparameter 
(including architecture parameters) tuning, to achive the result comparable with GBDT.

TODO: Systematic experiment execution and report here

## Execute

### Install
1. Create venv and install dependencies: 
    ```shell
    poetry install
    ```

### Prepare data
1. ```shell
    chmod +x ./scripts/data/*
    ```
2. Download data:
    ```shell
    ./scripts/data/download.sh ./data
    ```

### Train
The configuration is managed by hydra framework.
The default configuration is stored under `./configs`.

1. Activate venv
    ```
    poetry shell
    ```
2. Launch train of gbdt model overriding directories: 
    ```
    python -m superconduct_tc_reg.train pipeline=xgb dir.base=$(pwd)
    ```

### Create end-to-end model
Create end-to-end model, which includes data preprocessing and back-scaling of target and store it to registry.
You will need run id and source model name. Default source model name for gbdt is `superconduct-gbdt:onnx`

```
python -m superconduct_tc_reg.deploy.create_ete_model run_id=DESIRED_RUN_ID source_model_name="superconduct-gbdt:onnx"
```

### Deploy
Deploy to a local server as FastAPI service:
```
mlflow models serve -m "runs:/<run_id>/superconduct-gbdt:onnx:ete" -p 5000
```

## References

- Kam Hamidieh, . "A data-driven statistical model for predicting the critical temperature of a superconductor".Computational Materials Science (2018).

- Yu, Jiahao, Yongman, Zhao, Rongshun, Pan, Xue, Zhou, and Zikai, Wei. "Prediction of the Critical Temperature of Superconductors Based on Two-Layer Feature Selection and the Optuna-Stacking Ensemble Learning Model".ACS Omega 8, no.3 (2023): 3078-3090.