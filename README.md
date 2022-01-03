<span style="font-family: 'Times New Roman'">

# **Convex Clustering Optimization Problem**

This problem is the final project for the lecture Optimization for MDS in CUHKSZ. 

**Thank you, Andre!**  

## **Problem Description**

#### General Model
Let $\mathbb{R}^{d \times n} \ni A = (a_{1},a_{2},\dots,a_{n})$ be a given data matrix with observations and $d$ features. The convex clustering model for these $n$ solves the following convex optimization probelm:

$$
    \min_{x \in \mathbb{R}^{d \times n}} {1 \over 2} \sum_{i=1}^{n} \lVert x_{i} - a_{i} \rVert ^{2} + \lambda \sum_{i=1}^{n} \sum_{j=i+1}^{n} \lVert x_{i} - x_{j} \rVert_{p}
$$

where:
-  $\lambda > 0$ is a regularization paramter  
-  $\lVert \cdot \rVert_{p}$ is the standard Eucilidean norm
-  $X^{*} = (x_{1}^{\*}, \dots, x_{n}^{\*})$ is the optimal solution]


then $a_{i}$ and $a_{j}$ will be assigned as the same cluster iff.  $x_{i}^{\*} = x_{j}^{\*}$ or $\lVert x_{i}^{\*} - x_{j}^{\*}\rVert \le \epsilon$

#### Huber-type Norm
we choose $p = 2$ and we first consider a smooth variant of the clustering problem: 
$$
    \min_{x \in \mathbb{R}^{d \times n}} f_{cluster}(X) := {1 \over 2} \sum_{i=1}^{n} \lVert x_{i} - a_{i} \rVert ^{2} + \lambda \sum_{i=1}^{n} \sum_{j=i+1}^{n} \varphi_{hub}  (x_{i} - x_{j} ) _{p}
$$

where:

$$
     \varphi_{hub}(y) = \begin{cases}
                            {{1\over{2\delta}}{\lVert y \rVert^{2}}} &\text if {\rVert y \lVert \le \delta} \\\\
                            {\lVert y \rVert - {\delta \over{2}}}& \text if {\rVert y \lVert > \delta}
                        \end{cases}
$$


By some derivation, we can have the gradient of the Huber-norm function:

$$
   \nabla \varphi_{hub}(y) = \begin{cases}
                                {t / {\delta}} &\text if {\rVert y \lVert \le \delta} \\\\
                                { {y / {\lVert y \rVert}}}& \text if {\rVert y \lVert > \delta}
                            \end{cases}
$$

#### Matrix Improvement 
Main difficulties lie in the 2nd term in the objective function. 

In order to speed up the computing in each iteration, we can rewrite the gradient and Hessian in the form of matrix calculation.
##### (Gradient Matrix Expression) 
The full matrix expression of the gradient can be write down below as $G:$ 


$$
G = 
  \left(\begin{array}{c} \tiny 
+\nabla \varphi_{\text {hub }}\left(x_{1}-x_{1}\right)+\nabla \varphi_{\text {hub }}\left(x_{1}-x_{2}\right)+\nabla \varphi_{\text {hub }}\left(x_{1}-x_{3}\right)+\cdots+\nabla \varphi_{\text {hub }}\left(x_{1}-x_{n}\right) \\\\

\tiny -\nabla \varphi_{\text {hub }}\left(x_{1}-x_{2}\right)+\nabla \varphi_{\text {hub }}\left(x_{2}-x_{2}\right)+\nabla \varphi_{\text {hub }}\left(x_{2}-x_{3}\right)+\cdots\tiny+\nabla \varphi_{\text {hub }}\left(x_{2}-x_{n}\right) \\\\

\tiny -\nabla \varphi_{\text {hub }}\left(x_{1}-x_{3}\right)-\nabla \varphi_{\text {hub }}\left(x_{2}-x_{3}\right)+\nabla \varphi_{\text {hub }}\left(x_{3}-x_{3}\right)+\cdots+\nabla \varphi_{\text {hub }}\left(x_{3}-x_{n}\right) \\\\

\tiny \vdots \\\\

\tiny -\nabla \varphi_{\text {hub }}\left(x_{1}-x_{n}\right)-\nabla \varphi_{\text {hub }}\left(x_{2}-x_{n}\right)-\nabla \varphi_{\text {hub }}\left(x_{3}-x_{n}\right)-\cdots+\nabla \varphi_{\text {hub }}\left(x_{n}-x_{n}\right)
\end{array}\right)$$

$$  
\scriptsize G = 
 C(\nabla \varphi_{\text {hub}}\left(x_{1}-x_{2}\right) \; \nabla \varphi_{\text{hub}}\left(x_{1}-x_{3}\right) \dotsm \nabla \varphi_{\text{hub}}\left(x_{2}-x_{3}\right) \dotsm \nabla \varphi_{\text{hub}}\left(x_{n-1}-x_{n}\right))^{T}
$$

Noted that, if we calculate the full gradient, the huber norm should be computed in pairwise, which should be written in two **for loop**. 

To speed up this process, the gradient can be written in the following multiplation of matrixes: 
$$G = \nabla^{2}\varphi(
\scriptsize
\left(\begin{array}{cccccccc}
1 & -1 & 0 & 0 & 0 & \cdots & 0 & 0 \\\\
1 & 0 & -1 & 0 & 0 & \cdots & 0 & 0 \\\\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\\
1 & 0 & 0 & 0 & 0 & \cdots & 0 & -1 \\\\
0 & 1 & -1 & 0 & 0 & \cdots & 0 & 0 \\\\
0 & 1 & 0 & -1 & 0 & \cdots & 0 & 0 \\\\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\\
0 & 0 & 0 & 0 & 0 & \cdots & 1 & -1
\end{array}\right)\left(\begin{array}{c}
x_{1} \\\\
x_{2} \\\\
x_{3} \\\\
\vdots \\\\
x_{n-1} \\\\
x_{n}
\end{array}\right))
$$

##### (Hessian Matrix Expression) 
In Newton method, it is necessary for us to calculate $Hp$ to find the conjugate directions,however, it is very time-consuming to calculate the Hessian values by values. To speed up the
Newton iteration, we also figure out how to express it usingmatrix operations. In the experiments, it was shown that the matrix calculation can save around 50% time in each iteration.

Similarly, we first give the detailed format of hessian matrix of the second item which is: 

$$
H=\tiny \left(\begin{array}{cccc}
\sum_{j=1}^{n} \nabla^{2} \varphi\left(x_{1}-x_{j}\right) & -\nabla^{2} \varphi\left(x_{1}-x_{2}\right) & \cdots & -\nabla^{2} \varphi\left(x_{1}-x_{n}\right) \\\\
-\nabla^{2} \varphi\left(x_{1}-x_{2}\right) & \sum_{j=2}^{n} \nabla^{2} \varphi\left(x_{2}-x_{j}\right)+\nabla^{2} \varphi\left(x_{1}-x_{2}\right) & \cdots & -\nabla^{2} \varphi\left(x_{2}-x_{n}\right) \\\\
\vdots & \vdots & \ddots & \vdots \\\\
-\nabla^{2} \varphi\left(-x_{n}+x_{1}\right) & -\nabla^{2} \varphi\left(-x_{n}+x_{2}\right) & \cdots & \sum_{j=1}^{n} \nabla^{2} \varphi\left(x_{j}-x_{n}\right)
\end{array}\right)
$$


It is difficult to express it as the product of two matrices so we can express it as the sum of two matrices: 
$$
H = \nabla^{2}\varphi(diag(JB) + F)
$$

As for the $JB$, the matrix can be expressed as: 

$$
JB= \scriptsize
\begin{pmatrix}
	0 & 1 & 1 & \cdots & 1 \\\\
	1 & 0 & 1 & \cdots & 1 \\\\
	1 & 1 & 0 & \cdots & 1 \\\\ 
	\vdots & \vdots & \vdots & \ddots & \vdots \\\\
	1 & 1 & 1 & \cdots & 0
\end{pmatrix}
\begin{pmatrix}
	x_{1}-x_{1} & x_{1}-x_{2} & x_{1}-x_{3} & \cdots & x_{1}-x_{n} \\\\ 
	x_{1}-x_{2} & x_{2}-x_{2} & x_{2}-x_{3} & \cdots & x_{2}-x_{n} \\\\  
	x_{1}-x_{3} & x_{2}-x_{3} & x_{3}-x_{3} & \cdots & x_{3}-x_{n} \\\\ 
	\vdots & \vdots & \vdots & \ddots & \vdots \\\\
	x_{1}-x_{n} & x_{2}-x_{n} & x_{3}-x_{n} & \cdots & x_{n}-x_{n}
\end{pmatrix}
$$
Thus, $\nabla^{2}\varphi(diag(JB))=diag(H)$.  

And for the $F$, it is equal to:
$$
(-B + diag(B))
$$
While the calculation of $B$ is the same as the processing of $G$ in the former section **Gradient Matrix Expression**.

    The performance in our code shows that the matrices 
    processing can reduce the CPU time by 50% in each iteration.

#### Accerlated Gradient Method

**Algorithm of AGM**
   1. Initialization: Choose a point $x^{0}$ and set $x^{-1} = x^{0}$
   2. Select $\beta_{k}$ and compute the step $y^{k+1} = x^{k}+\beta_{k}(x^{k} - x^{k-1})$
   3. Select a step size $\alpha_{k}>0$ and set $x^{k+1} = y^{k+1} - \alpha \nabla f(y^{k+1})$

Here we choose constant stepsize with parameters:
$$alpha_{k} = {1 \over L}, \beta_{k} = {{t_{k-1}-1} \over t_{k}}$$
$$t_{k}=\scriptsize{1 \over 2}(1 + \sqrt{1+4t_{k-1}^{2}}), \normalsize t_{-1} = t_{0}=1$$



#### Newton-CG Method

**Algorithm of NCG**
   1. Initialization: Choose a point $x^{0}$ and choose $\sigma, \gamma \in (0,1), and (\rho_{k})_{k}, k \in \mathbb{N}$.
   2. Set $A = \nabla^{2}f(x^{k}), v^{0}=0, r^{0}=\nabla f(x^{k}), p^{0} = -r^{0}$ and $tol = \rho_{k}\lVert \nabla f(x^{k})\rVert$.
   3. For $j =0,1,\dotsm$
       * if $(p^{j})^{\top}Ap^{j} \le 0$ return $d^{k} = v^{j}$
       * compute $\sigma_{j} = {{\lVert r^{j} \rVert^{2}} \over {(p^{j})^{\top}Ap^{j}} }$ , $v^{j+1} = v^{j} + \sigma_{j}p^{j}, r^{j+1} = r^{j} + \sigma_{j}Ap^{j}.$
       * If $\lVert r^{j+1} \rVert \le \text{tol}$ then **STOP** and return $d^{k} = v^{j+1}.$ 
       * Calculate $\beta_{j+1} = {{\lVert r^{j+1} \rVert}^{2} \over {\lVert r^{j} \rVert}^{2}}$ and $p^{j+1}=-r^{j+1}+\beta_{j+1}p^{j}
   4. Calculate $\alpha_{k}$ via backtracking and update $x^{k}$
   5. If gradient less than $\epsilon$, then STOP and output.


</span>


