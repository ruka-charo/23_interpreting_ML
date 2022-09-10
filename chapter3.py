# docker環境からフォルダへ
import os
os.chdir('work/23_interpreting_ML')

#%% インポート
import sys
import warnings
from dataclasses import dataclass
from typing import Any  # 型ヒント用
from __future__ import annotations  # 型ヒント用

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # matplotlibの日本語表示対応

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 自作モジュール
sys.path.append("..")
from mli.visualize import get_visualization_setting
from chapter3_pfi import *

np.random.seed(42)
pd.options.display.float_format = "{:.2f}".format
sns.set(**get_visualization_setting())
warnings.simplefilter("ignore")  # warningsを非表示に


#%% シミュレーションデータも訓練データとテストデータを分けたいので
def generate_simulation_data(N, beta, mu, Sigma):
    """線形のシミュレーションデータを生成し、訓練データとテストデータに分割する

    Args:
        N: インスタンスの数
        beta: 各特徴量の傾き
        mu: 各特徴量は多変量正規分布から生成される。その平均。
        Sigma: 各特徴量は多変量正規分布から生成される。その分散共分散行列。
    """

    # 多変量正規分布からデータを生成する
    X = np.random.multivariate_normal(mu, Sigma, N)

    # ノイズは平均0標準偏差0.1(分散は0.01)で決め打ち
    epsilon = np.random.normal(0, 0.1, N)

    # 特徴量とノイズの線形和で目的変数を作成
    y = X @ beta + epsilon

    return train_test_split(X, y, test_size=0.2, random_state=42)


# シミュレーションデータの設定
N = 1000
J = 3
mu = np.zeros(J)
Sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
beta = np.array([0, 1, 2])

# シミュレーションデータの生成
X_train, X_test, y_train, y_test = generate_simulation_data(N, beta, mu, Sigma)


#%% 散布図の作成
def plot_scatter(X, y, var_names):
    """目的変数と特徴量の散布図を作成"""

    # 特徴量の数だけ散布図を作成
    J = X.shape[1]
    fig, axes = plt.subplots(nrows=J, ncols=1, figsize=(4, 4*J))

    for d, ax in enumerate(axes):
        sns.scatterplot(X[:, d], y, alpha=0.3, ax=ax)
        ax.set(
            xlabel=var_names[d],
            ylabel="Y",
            xlim=(X.min() * 1.1, X.max() * 1.1)
        )

    fig.show()


# 可視化
var_names = [f"X{j}" for j in range(J)]
plot_scatter(X_train, y_train, var_names)



#%% 線形回帰モデルの特徴量重要度の確認
def plot_bar(variables, values, title=None, xlabel=None, ylabel=None):
    """回帰係数の大きさを確認する棒グラフを作成"""

    fig, ax = plt.subplots()
    ax.barh(variables, values)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, None))
    fig.suptitle(title)

    fig.show()


# 線形回帰モデルの学習
lm = LinearRegression().fit(X_train, y_train)

# 回帰係数の可視化
plot_bar(var_names, lm.coef_, "線形回帰の回帰係数の大きさ", "回帰係数")


#%% PFIのシミュレーションデータへの適用
# Random Forestの予測モデルを構築
rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)

# 予測精度を確認
print(f"R2: {r2_score(y_test, rf.predict(X_test)):.2f}")


#%% PFIを計算して可視化
# PFIのインスタンスを作成
pfi = PermutationFeatureImportance(rf, X_test, y_test, var_names)

# PFIを計算
pfi.permutation_feature_importance()

# PFIを可視化
pfi.plot(importance_type='difference')
