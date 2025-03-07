from pathlib import Path
from typing import Literal
import hydra
from omegaconf import OmegaConf
import pandas as pd
import sklearn.decomposition
import sklearn.preprocessing
import numpy as np

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


def handle_group_of_duplicates(group: pd.DataFrame, threshold: int, col_target: str):
    group_size = len(group)

    if group_size == 1:
        return pd.Series({col_target: group[col_target].mean()})
    elif group_size == 2:
        mag = group[col_target].max() - group[col_target].min()
        # We can't decide, which one is correct => drop both
        if mag >= threshold:
            return pd.Series({col_target: None})
        else:
            return pd.Series({col_target: group[col_target].mean()})
    else:
        # Return median
        return pd.Series({col_target: group[col_target].median()})


def remove_duplicates(
    df: pd.DataFrame, df_elements: pd.DataFrame, threshold: float, col_target: str
):
    df_elements_deduplicated = (
        df_elements.groupby("material", as_index=True)
        .apply(handle_group_of_duplicates, threshold=threshold, col_target=col_target)  # type: ignore
        .dropna(subset=[col_target])
    )
    df = (
        df.assign(**{"material": df_elements["material"]})
        .drop_duplicates(subset=["material"], keep="first")
        .set_index("material")
        .filter(items=df_elements_deduplicated.index, axis=0)
    )
    df[col_target] = df_elements_deduplicated[col_target]
    return df, df_elements_deduplicated


def remove_outliers(df: pd.DataFrame, df_elements: pd.DataFrame, outliers: list[int]):
    return df.drop(index=outliers), df_elements.drop(index=outliers)


@hydra.main(
    config_path="../../../configs",
    config_name="preprocess",
    version_base="1.3",
)
def process_data(config):

    df = pd.read_csv(config["dataset"]["dir"])
    df_elements = pd.read_csv(config["dataset"]["dir_elements"])

    steps = config["preprocessing"]["steps"]

    if "outliers" in steps:
        print("Handle outliers ...")
        df, df_elements = hydra.utils.instantiate(steps["outliers"])(df, df_elements)

    if "duplicates" in steps:
        print("Handle duplicated ...")
        df, df_elements = hydra.utils.instantiate(steps["duplicates"])(df, df_elements)

    df_features = df.drop(config["target"], axis=1)

    if "scale" in steps:
        print("Scale ...")
        scaler = hydra.utils.instantiate(steps["scale"])
        df_features = pd.DataFrame(
            scaler.fit_transform(df_features),
            columns=df_features.columns,
            index=df.index,
        )

    if "CFS" in steps:
        print("CFS ...")
        df_features = hydra.utils.instantiate(steps["CFS"])(df_features)
        print(f"Selected features: {df_features.columns}, N_f={len(df_features.columns)}")

    if "pca" in steps:
        print("PCA ...")
        pca: sklearn.decomposition.PCA = hydra.utils.instantiate(steps["pca"])
        df_features = pd.DataFrame(
            pca.fit_transform(df_features),
            columns=[f"pc_{i}" for i in range(pca.n_components_)],
            index=df.index,
        )

    df_features[config["target"]] = df[config["target"]]
    assert not df_features[config["target"]].isna().any()

    if config["preprocessing"]["do_save"]:
        out_path = Path(config["out_file_path"])
        out_path.parent.mkdir(exist_ok=True, parents=True)
        df_features.to_csv(out_path, index=False)

    return df_features


if __name__ == "__main__":
    process_data()
