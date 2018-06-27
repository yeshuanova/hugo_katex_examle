---
title: "Multidimensional Scaling"
author: "Cheng-Shiang Li"
description: "Multidimensional Scaling"
date: "2017-08-10T20:00:00Z"
categories:
  - "Dimension Reduction"
tags:
  - "MDS"
  - "Python"
markup: mmark
---

在[之前]({{ site.baseurl }}/posts/dimension-reduction-tutorial/)已經介紹過資料降維的基礎概念
以及使用進行PCA示範，因此這篇要來談的是另一種降維方式
[MDS](https://en.wikipedia.org/wiki/Multidimensional_scaling)，並敘述相關理論推導。

<!--more-->

## Introduction

MDS全名為**Multidimensional Scaling**，和之前使用過的PCA一樣是非常經典的演算法，
現在也衍生如出許多進階版本，在本篇我們將主要介紹放在古典MDS演算法內容。

MDS主要概念為透過保持資料間**歐式距離（Euclidean distance）**關係，
也就是希望轉換後的低維資料中每筆資料間的**距離**，盡可能和高維資料的資料間距離保持差異最小化。
假設現在有一存在$$n$$筆資料的資料集$$A=\{a_1,a_2,\ldots,a_n\}$$，經過MDS轉換後成為新資料集
$$B=\{b_1,b_2,\dots,b_n\}$$，則

$$d(a_i,a_j) \approx d(b_i,b_j)$$

兩點間的歐式距離$$d(x, y)$$為

$$d(x, y) = \sqrt{(x-y)(x-y)^T}$$

## Matrix Form

假如現在有一資料集$$X$$並將其表示為$$m \times n$$矩陣，其中$$m$$為資料數量，$$n$$為資料維度，則矩陣表示式如下：

$$X=\begin{bmatrix} M_1 \\ \vdots \\ M_n \end{bmatrix}$$

$$X=\begin{bmatrix}X_1 \\ \vdots \\ X_m\end{bmatrix}=\begin{bmatrix}x_{11} & \cdots & x_{1n} \\ \vdots & & \vdots \\ x_{m1} & \cdots & x_{mn}\end{bmatrix}$$

而降維後的資料矩陣 $$Z$$ 為 $$m \times k$$ 與 $$k \leq n$$，則

$$Z=\begin{bmatrix} Z_1 \\ \vdots \\ X_m \end{bmatrix} = \begin{bmatrix} z_{11} & \cdots & z_{1k} \\ \vdots & & \vdots \\ z_{m1} & \cdots & z_{mk} \end{bmatrix}$$

定義$$X$$與$$Z$$的$$m \times m$$歐式距離矩陣為$$D$$與$$P$$，其中

$$D_{ij}=d(X_i, X_j)$$

$$P_{rs}=d(Z_r, Z_s)$$

假定$$Z$$中所有資料相加為零，即$$\sum\limits_{r}{}Z_r=0$$，經推導後$$Z$$的內積矩陣$$B$$可轉換為

$$B=-\frac{1}{2}HDH=ZZ^T$$

其中

$$H=I-\frac{1}{m}11^{'}$$

對$$B$$進行[特徵根分解](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix)可得

$$B=V\Lambda V^T=(V\Lambda^{\frac{1}{2}})(V\Lambda^{\frac{1}{2}})^T=ZZ^T$$

由上式可知

$$Z=V\Lambda^{\frac{1}{2}}$$

其中$$\Lambda$$為$$B$$的特徵根對角矩陣，$$\Lambda_{ii}=\lambda_i$$。$$V$$為特徵向量矩陣
，$$V=[v_1, v_2,\dots,v_m]$$。且
$$\lambda_1 \geq \lambda_2 \dots \lambda_m \geq 0$$

## Algorithm

由以上的推導可知，降維資料$$Z$$可由以下步驟加以求得

- 計算原始資料距離矩陣$$D=[d_ij]$$

- 轉換爲Centering Matrix $$B=-\frac{1}{2}HDH$$，其中$$H=I-\frac{1}{m}11^{'}$$

- 求出$$B$$的最大$$k$$個特徵根$$\lambda_{1}, \lambda _{2},...,\lambda _{k}$$，與其對應的特徵向量$$e_{1},e_{2},..., e_{k}$$

- 因此$$Z$$可由$$Z=E_{k}\Lambda_{k}^{\frac{1}{2}}$$求得，
  其中$$\Lambda_{k}^{\frac{1}{2}}$$為$$k \times k$$的特徵根對角矩陣，而
  $$E_{k}$$為對應的$$m \times k$$特徵向量矩陣。

## Code

以下為將Iris flower dataset與scikit-learn套件，進行MDS降維與結果展示

```python
# 匯入相關 Package
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import MDS

# 取得 Iris data set
iris = datasets.load_iris()
iris_label = iris.target

# 設定 MDS Model參數並對 dataset進行降維
mds_coeff = MDS(n_components=2)
mds_result = mds_coeff.fit_transform(iris.data)

# 繪製資料2D表示圖，並將不同類別加上顏色標示
for idx, label in enumerate(iris_label):
    if label == 0:
        color = 'ro'
    elif label == 1:
        color = 'go'
    elif label == 2:
        color = 'bo'
    else:
        continue
    x = mds_result[idx, 0]
    y = mds_result[idx, 1]
    plt.plot(x, y, color)
```

下圖為執行結果，可看data set中的各類別，在降低到2維仍有明確的群聚關係。

![Iris-MDS-2d](/images/multi-dimensional-scaling/iris_mds_2d.svg)