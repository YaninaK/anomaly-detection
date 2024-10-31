import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

__all__ = ["make_pca_transformations"]


def make_pca_transformations(
    seq_features: np.array,
    stat_features: np.array,
    df_stat: pd.DataFrame,
    period: int,
) -> Tuple[pd.DataFrame, np.array]:
    """
    1. Определяет необходимое число компонентов для PCA.
    2. Проводит PCA трансформацию.
    3. Рассчитывает Hotelling's T-squared и Q residuals для определения аномалий.
    """
    t = seq_features[:, period]
    X = np.hstack([stat_features[t > 0], np.expand_dims(t[t > 0], axis=1)])

    corr = np.corrcoef(X, rowvar=False)
    pca_estimate = PCA(n_components="mle")
    pca_estimate.fit(corr)
    n_components = (
        pca_estimate.explained_variance_ratio_
        > 1 / len(pca_estimate.explained_variance_ratio_)
    ).sum()

    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    cols = [f"factor_{i}" for i in range(n_components)]
    df = pd.DataFrame(X_transformed, columns=cols)

    lambda_inv = linalg.inv(
        np.dot(X_transformed.T, X_transformed) / (X_transformed.shape[0] - 1)
    )
    df["Hotelling's T-squared"] = df.T.apply(
        lambda t: np.dot(np.dot(t, lambda_inv), t.T)
    )
    errors = X - np.dot(X_transformed, pca.components_)
    df["Q residuals"] = pd.DataFrame(errors.T).apply(lambda e: np.dot(e, e.T))

    return df, X
