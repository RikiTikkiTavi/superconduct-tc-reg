from pathlib import Path
from typing import Literal
import hydra
from omegaconf import OmegaConf
import pandas as pd
import sklearn.decomposition
import sklearn.preprocessing
import numpy as np

ScalerType = sklearn.preprocessing.StandardScaler | sklearn.preprocessing.MinMaxScaler


def remove_highly_correlated_features(
    df: pd.DataFrame, threshold=0.95, method: Literal["pearson", "spearman"] = "pearson"
):
    """
    Removes highly correlated features from a dataset.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr(method=method).abs()

    # Select the upper triangle of the correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation greater than the threshold
    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    # Drop the highly correlated features
    df_dropped = df.drop(columns=to_drop)

    return df_dropped


def process_data():
    config = OmegaConf.load("params.yaml")
    df = pd.read_csv(config["dataset"]["dir"])
    df_features = df.drop(config["target"], axis=1)

    if config["scaler"] is not None:
        scaler: ScalerType = hydra.utils.instantiate(config["scaler"])
        df_features = pd.DataFrame(
            scaler.fit_transform(df_features), columns=df_features.columns
        )

    if config["CFS"] is not None:
        df_features = hydra.utils.instantiate(config["CFS"])(df_features)

    if config["PCA"] is not None:
        pca: sklearn.decomposition.PCA = hydra.utils.instantiate(config["PCA"])
        df_features = pd.DataFrame(
            pca.fit_transform(df_features),
            columns=[f"pc_{i}" for i in range(pca.n_components_)],
        )

    out_path = Path(config["out_path"])
    out_path.parent.mkdir(exist_ok=True, parents=True)

    df_features[config["target"]] = df[config["target"]]
    df_features.to_csv(config["out_path"])

    return df_features


if __name__ == "__main__":
    process_data()
