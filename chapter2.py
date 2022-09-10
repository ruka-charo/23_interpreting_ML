# docker環境からフォルダへ
import os
os.chdir('work/23_interpreting_ML')

#%% インポート
import sys
import warnings
from dataclasses import dataclass
from typing import Any
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# 自作モジュール
sys.path.append('..')
from mli.visualize import get_visualization_setting

np.random.seed(42)
pd.options.display.float_format = '{:.2f}'.format
sns.set(**get_visualization_setting())
warnings.simplefilter('ignore') # warningsを非表示に


#%% データセットの読み込み
boston = load_boston()

# データセットはdictで与えられる
# dataに特徴量が、targetに目的変数が格納されている
X = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
y = boston['target']


#%% データの前処理
def plot_histgram(x, title=None, x_label=None):
    '与えられた特徴量のヒストグラムを作成'

    fig, ax = plt.subplots()
    sns.distplot(x, kde=False, ax=ax)
    ax.set_xlabel(x_label)

    fig.show()


plot_histgram(y, title='目的変数の分布', x_label='MEDV')

#%%
def plot_scatters(X, y, title=None):
    """目的変数と特徴量の散布図を作成"""

    cols = X.columns
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for ax, c in zip(axes.ravel(), cols):
        sns.scatterplot(X[c], y, ci=None, ax=ax)
        ax.set(ylabel="MEDV")

    fig.suptitle(title)

    fig.show()


plot_scatters(
X[['RM', 'LSTAT', 'DIS', 'CRIM']],
y,
title='目的変数と各特徴量の関係'
)


#%% 線形モデルの学習と評価
# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 後で使えるようにデータを書き出しておく
joblib.dump(
    [X_train, X_test, y_train, y_test],
    filename='data/boston_housing.pkl'
)

#%% 学習
lm = LinearRegression()
lm.fit(X_train, y_train)


#%% 評価指標の計算
def regression_metrics(estimator, X, y):
    """回帰精度の評価指標をまとめて返す関数"""

    # テストデータで予測
    y_pred = estimator.predict(X)

    # 評価指標をデータフレームにまとめる
    df = pd.DataFrame(
        data={
            "RMSE": [mean_squared_error(y, y_pred, squared=False)],
            "R2": [r2_score(y, y_pred)],
        }
    )

    return df

# 精度評価
regression_metrics(lm, X_test, y_test)


#%% 回帰係数
def get_coef(estimator, var_names):
    """特徴量名と回帰係数が対応したデータフレームを作成する"""

    # 切片含む回帰係数と特徴量の名前を抜き出してデータフレームにまとめる
    df = pd.DataFrame(
        data={"coef": [estimator.intercept_] + estimator.coef_.tolist()},
        index=["intercept"] + var_names
    )

    return df

df_coef = get_coef(lm, X.columns.tolist())
df_coef


#%% 元のデータを上書きしないようにコピーしておく
X_train2 = X_train.copy()
X_test2 = X_test.copy()

# 2乗項を追加
X_train2['LSTAT2'] = X_train2['LSTAT'] ** 2
X_test2['LSTAT2'] = X_test2['LSTAT'] ** 2

# 学習
lm2 = LinearRegression()
lm2.fit(X_train2, y_train)

# 精度評価
regression_metrics(lm2, X_test2, y_test)

# 二乗項を追加した場合の回帰係数を出力
df_coef2 = get_coef(lm2, X_train2.columns.tolist())
df_coef2


#%% 標準化
ss = StandardScaler()

X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.fit_transform(X_test)

# 学習
lm_ss = LinearRegression()
lm_ss.fit(X_train_ss, y_train)

# 精度評価
regression_metrics(lm_ss, X_test_ss, y_test)

# 回帰係数
df_coef_ss = get_coef(lm_ss, X_train.columns.tolist())
df_coef_ss
