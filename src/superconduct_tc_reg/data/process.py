from pathlib import Path
from typing import Literal
import hydra
from omegaconf import OmegaConf
import pandas as pd
import sklearn.decomposition
import sklearn.preprocessing
import numpy as np

from superconduct_tc_reg.data.target_scaler import TargetScaler


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


class DataProcessor:
    original_features: list[str] | None = None
    target_scaler: TargetScaler | None = None
    example: pd.DataFrame | None = None

    def __init__(self, config):
        self._config = config

    def transform_features(self, df_features: pd.DataFrame):
        config = self._config
        steps = config["preprocessing"]["steps"]

        if "scale" in steps:
            print("Scale ...")
            df_features = pd.DataFrame(
                self._scaler.transform(df_features),
                columns=df_features.columns,
                index=df_features.index,
            )

        if "CFS" in steps:
            print("CFS ...")
            df_features = df_features[self._features]

        if "pca" in steps:
            print("PCA ...")
            df_features = pd.DataFrame(
                self._pca.transform(df_features),
                columns=[f"pc_{i}" for i in range(self._pca.n_components_)],
                index=df_features.index,
            )

        return df_features

    def fit_transform(
        self, df_features: pd.DataFrame, df_elements: pd.DataFrame, target: pd.Series
    ) -> pd.DataFrame:

        config = self._config

        steps = config["preprocessing"]["steps"]

        df = df_features.assign(**{config["target"]: target})
        self.original_features = df_features.columns.to_list()
        self.example = df_features.head(1)

        if "outliers" in steps:
            print("Handle outliers ...")
            df, df_elements = hydra.utils.instantiate(steps["outliers"])(
                df, df_elements
            )

        if "duplicates" in steps:
            print("Handle duplicated ...")
            df, df_elements = hydra.utils.instantiate(steps["duplicates"])(
                df, df_elements
            )

        df_features = df.drop(config["target"], axis=1)

        if "scale" in steps:
            print("Scale ...")
            scaler = hydra.utils.instantiate(steps["scale"])
            df_features = pd.DataFrame(
                scaler.fit_transform(df_features),
                columns=df_features.columns,
                index=df.index,
            )
            self._scaler = scaler

        if "CFS" in steps:
            print("CFS ...")
            df_features = hydra.utils.instantiate(steps["CFS"])(df_features)
            self._features = df_features.columns.to_list()
            print(
                f"Selected features: {df_features.columns}, N_f={len(df_features.columns)}"
            )

        if "pca" in steps:
            print("PCA ...")
            pca: sklearn.decomposition.PCA = hydra.utils.instantiate(steps["pca"])
            df_features = pd.DataFrame(
                pca.fit_transform(df_features),
                columns=[f"pc_{i}" for i in range(pca.n_components_)],
                index=df.index,
            )
            self._pca = pca

        if "target_scaler" in steps:
            target_scaler = hydra.utils.instantiate(steps["target_scaler"])
            df[config["target"]] = target_scaler.fit_transform(df[config["target"]])  # type: ignore
            self.target_scaler = target_scaler

        df_features[config["target"]] = df[config["target"]]
        assert not df_features[config["target"]].isna().any()

        return df_features

    def read_fit_transform(self) -> pd.DataFrame:
        df = pd.read_csv(self._config["dataset"]["dir"])
        df_elements = pd.read_csv(self._config["dataset"]["dir_elements"])
        df_features = df.drop(self._config["target"], axis=1)
        return self.fit_transform(df_features, df_elements, df[self._config["target"]])


@hydra.main(
    config_path="../../../configs",
    config_name="preprocess",
    version_base="1.3",
)
def process_data(config):
    data_processor = DataProcessor(config)
    df = data_processor.read_fit_transform()

    if config["preprocessing"]["do_save"]:
        out_path = Path(config["out_file_path"])
        out_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(out_path, index=False)

    return df


if __name__ == "__main__":
    process_data()
