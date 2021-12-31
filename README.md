<span style="font-family: 'Times New Roman'">


# **Convex Clustering Optimization Problem**

This problem is the final project for the lecture Optimization for MDS in CUHKSZ. 
Anyway, Thank you, Andre!  

## **Problem Description**

#### General Model
Let $\mathbb{R}^{d \times n} \ni A = (a_{1},a_{2},\dots,a_{n})$ be a given data matrix with observations and $d$ features. The convex clustering model for these $n$ solves the following convex optimization probelm:

$$
    \min_{x \in \mathbb{R}^{d \times n}} {1 \over 2} \sum_{i=1}^{n} \lVert x_{i} - a_{i} \rVert ^{2} + \lambda \sum_{i=1}^{n} \sum_{j=i+1}^{n} \lVert x_{i} - x_{j} \rVert_{p}
$$

where:
-  $\lambda >0$ is a regularization paramter  
-  $\lVert \cdot \rVert _{p}$ is the standard Eucilidean norm
- $X^{*} = (x_{1}^{*}, \dots, x_{n}^{*})$ is the optimal solution

then $a_{i}$ and $a_{j}$ will be assigned as the same cluster iff.  $x_{i}^{*} = x_{j}^{*}$ or $\lVert x_{i}^{*} - x_{j}^{*}\rVert \le \epsilon $

#### Huber-type Norm
we choose $p = 2$ and we first consider a smooth variant of the clustering problem: 
$$
    \min_{x \in \mathbb{R}^{d \times n}} f_{cluster}(X) := {1 \over 2} \sum_{i=1}^{n} \lVert x_{i} - a_{i} \rVert ^{2} + \lambda \sum_{i=1}^{n} \sum_{j=i+1}^{n} \varphi_{hub}  (x_{i} - x_{j} ) _{p}
$$

where:

$$
     \varphi_{hub}(y) = \begin{cases}
                            {{1\over{2\delta}}{\lVert y \rVert^{2}}} &\text if {\rVert y \lVert \le \delta} \\
                            {\lVert y \rVert - {\delta \over{2}}}& \text if {\rVert y \lVert > \delta}
                        \end{cases}
$$


By some derivation, we can have the gradient of the Huber-norm function:

$$
   \nabla \varphi_{hub}(y) = \begin{cases}
                                {t / {\delta}} &\text if {\rVert y \lVert \le \delta} \\
                                { {y / {\lVert y \rVert}}}& \text if {\rVert y \lVert > \delta}
                            \end{cases}
$$

#### Matrix Improvement 
In order to speed up the computing in each iteration, we can rewrite the gradient and Hessian in the form of matrix calculation.
##### Gradient Matrix Expression 


#### Accerlated Gradient Method

</span>

