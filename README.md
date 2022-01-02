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
-  $ \lambda >0 $ is a regularization paramter  
-  $ \lVert \cdot \rVert _{p} $ is the standard Eucilidean norm
-  $ X^{\*} = (x_{1}^{\*}, \dots, x_{n}^{\*}) $ is the optimal solution

then $a_{i}$ and $a_{j}$ will be assigned as the same cluster iff.  $x_{i}^{\*} = x_{j}^{\*}$ or $\lVert x_{i}^{\*} - x_{j}^{\*}\rVert \le \epsilon $

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
##### (Gradient Matrix Expression) 
The full matrix expression of the gradient can be write down below as $ G:$ 


$$ G = \tiny 
  \left(\begin{array}{c}
+\nabla \varphi_{\text {hub }}\left(x_{1}-x_{1}\right)+\nabla \varphi_{\text {hub }}\left(x_{1}-x_{2}\right)+\nabla \varphi_{\text {hub }}\left(x_{1}-x_{3}\right)+\cdots+\nabla \varphi_{\text {hub }}\left(x_{1}-x_{n}\right) \\
-\nabla \varphi_{\text {hub }}\left(x_{1}-x_{2}\right)+\nabla \varphi_{\text {hub }}\left(x_{2}-x_{2}\right)+\nabla \varphi_{\text {hub }}\left(x_{2}-x_{3}\right)+\cdots+\nabla \varphi_{\text {hub }}\left(x_{2}-x_{n}\right) \\
-\nabla \varphi_{\text {hub }}\left(x_{1}-x_{3}\right)-\nabla \varphi_{\text {hub }}\left(x_{2}-x_{3}\right)+\nabla \varphi_{\text {hub }}\left(x_{3}-x_{3}\right)+\cdots+\nabla \varphi_{\text {hub }}\left(x_{3}-x_{n}\right) \\
\vdots \\
-\nabla \varphi_{\text {hub }}\left(x_{1}-x_{n}\right)-\nabla \varphi_{\text {hub }}\left(x_{2}-x_{n}\right)-\nabla \varphi_{\text {hub }}\left(x_{3}-x_{n}\right)-\cdots+\nabla \varphi_{\text {hub }}\left(x_{n}-x_{n}\right)
\end{array}\right)$$

$$  
G = \scriptsize
 C(\nabla \varphi_{\text {hub}}\left(x_{1}-x_{2}\right) \; \nabla \varphi_{\text{hub}}\left(x_{1}-x_{3}\right) \dotsm \nabla \varphi_{\text{hub}}\left(x_{2}-x_{3}\right) \dotsm \nabla \varphi_{\text{hub}}\left(x_{n-1}-x_{n}\right))^{T}
$$

Noted that, if we calculate the full gradient, the huber norm should be computed in pairwise, which should be written in two **for loop**. 

To speed up this process, the gradient can be written in the following multiplation of matrixes: 
<!-- <img src="https://latex.codecogs.com/svg.image?\left(\begin{array}{cccccccc}1&space;&&space;-1&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;\cdots&space;&&space;0&space;&&space;0&space;\\1&space;&&space;0&space;&&space;-1&space;&&space;0&space;&&space;0&space;&&space;\cdots&space;&&space;0&space;&&space;0&space;\\\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;&&space;\vdots&space;\\1&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;\cdots&space;&&space;0&space;&&space;-1&space;\\0&space;&&space;1&space;&&space;-1&space;&&space;0&space;&&space;0&space;&&space;\cdots&space;&&space;0&space;&&space;0&space;\\0&space;&&space;1&space;&&space;0&space;&&space;-1&space;&&space;0&space;&&space;\cdots&space;&&space;0&space;&&space;0&space;\\\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;&&space;\vdots&space;\\0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;\cdots&space;&&space;1&space;&&space;-1\end{array}\right)\left(\begin{array}{c}x_{1}&space;\\x_{2}&space;\\x_{3}&space;\\\vdots&space;\\x_{n-1}&space;\\x_{n}\end{array}\right)" title="\left(\begin{array}{cccccccc}1 & -1 & 0 & 0 & 0 & \cdots & 0 & 0 \\1 & 0 & -1 & 0 & 0 & \cdots & 0 & 0 \\\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\1 & 0 & 0 & 0 & 0 & \cdots & 0 & -1 \\0 & 1 & -1 & 0 & 0 & \cdots & 0 & 0 \\0 & 1 & 0 & -1 & 0 & \cdots & 0 & 0 \\\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\0 & 0 & 0 & 0 & 0 & \cdots & 1 & -1\end{array}\right)\left(\begin{array}{c}x_{1} \\x_{2} \\x_{3} \\\vdots \\x_{n-1} \\x_{n}\end{array}\right)" /> -->


$$G = 
\small
\left(\begin{array}{cccccccc}
1 & -1 & 0 & 0 & 0 & \cdots & 0 & 0 \\
1 & 0 & -1 & 0 & 0 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
1 & 0 & 0 & 0 & 0 & \cdots & 0 & -1 \\
0 & 1 & -1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 1 & 0 & -1 & 0 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & 0 & \cdots & 1 & -1
\end{array}\right)\left(\begin{array}{c}
x_{1} \\
x_{2} \\
x_{3} \\
\vdots \\
x_{n-1} \\
x_{n}
\end{array}\right)
$$

##### (Hessian Matrix Expression) 


#### Accerlated Gradient Method

</span>

