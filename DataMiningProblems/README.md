# Data-Mining-Problems

本文档使用指南：

1. 注意本文档构成：我们copy的以往学长考试总结+YB老师上课录音听力+其他课程的公式推导

2. 本着互利分享的原则，你可以选择自行学习或者你可以选择贡献一部分力量：当你发现问题时，或者有更好的解答时候，可以对文档进行修改并pull request。
3. 我们建议你先学习Technical Problems。在保证你对以下technical problem有所孰知的时候，再阅读下面的问题，你将会更能理解YB老师的意思。

# Technical Problems

1. SVM formulations

2. PCA formulation, SVD formulation, and eigenvalue decomposition.

3. What is Sparse PCA? What is low-rank PCA?

   [Sparse Principal Component Analysis](<https://www.tandfonline.com/doi/pdf/10.1198/106186006X113430?needAccess=true>)

   [Sparse PCA through Low-rank Approximations](<http://proceedings.mlr.press/v28/papailiopoulos13.pdf>)

4. EM algorithm and its locality

5. HMM and $\alpha$- and $ \beta$-update 

6. Graphical model

7. Gaussian Mixture Model

8. Regressions

# Reference Problems

### Learning and Search

**1) Why all learning problems are inverse problems, requiring unbounded exhaustive searches, thus ill-posed?** 

Because learning problem is learning from data to get a general model which means it needs to learn general rules from few data

因为数据挖掘处理的数据往往有着维数高，多模型混合生乘，时变等复杂特征，当我们通过⼀个模型去拟合数据时，肯定忽略了数据的⼀些复杂特征，或者说， 不管我们采用哪种模型，都不可能保证完全形容出数据的所有特征。而且在我们选择模型的时候，也不可能从所有的模型中选择最好的，因为穷举的复杂度是极高的。

**2) Why gradient is the key mathematical assumption that we could count on in order to search? What would be the general implications for the existence of at least some continuality or locality?**

1. Gradient is the basis in Data Mining/AI/ML. 

    - Assumption : locally stationary, locally smooth (not jumping) ---- Locality.

    - Developing a gradient system enables us to tune the parameter following the direction of gradient. It save us from exhaustive search.

2. By gradient, we assume that the function is locally smooth, continuous, has stationary point, differentiable.

    - (Not necessarily global, but at least locally)The real world is likely to be continuous physically. Natural process is unlikely to be discrete.

    - Also, if no gradient exists, no other math tools we can rely on.

**3) What is the generalizability of a mathematical process, from both expressive (smoothness) and inclusive (capacity) point of views?**

Generalizability 普遍性 shows whether our model can achieve good result in new test dataset and is robust in solving real world problems.

1. From expressive point of view: the model/process can encode/decode/translate the input as detail as possible 细节处的表达效果好 ( avoid under-fitting)

2. From capacity point of view: it works on not only the given data but also all possible x&y(new data, test data) (avoid over-fitting) 不仅在已有的数据效果好，而且在其他新的数据效果好

**4) What would be some of the solutions for such an ill-posed problem in order to yield at least some reasonable results?**

- Gradient. Reduce dim. All kinds of tricks. Normalize. Sparse. Convex. Linear boundary. Try our best to deal with the most ill-posed problems.
- Soft margin: overlapping class Kernel method: mapping nonlinear relationship to linear one
    Regularization: dimension reduction to reduce complexity Cross validation: finding a model that generalizes well to testing data

**5) What are some of the mathematical hurdles that have prevented more generalizable solutions?** 有哪些数学障碍阻碍了更普遍的解决方案

- too linear and not expressive 模型太线性，涵盖模型空间小
- local solutions and Singularity. 局部解和奇点
- Increase dimension or Curse of dimensionality  维度增加或者维度的诅咒
- Heterogeneous /dirty data space 异构的/脏的 数据空间
- Difficult to project real (x,y,z) to Euclidean space 很难将真实空间投影到欧式空间
- Higher mathematical/computational complexity. 很高的数学/计算 复杂度

We have to find alternative/compromise/tricks to overcome.  必须找到 替代/妥协/技巧 来解决

**6) Why variable dependences (interactions) could become an extremely difficult and even an impossible problem? Give philosophical, mathematical, physical, computational, and numerical examples for such a singularity.**  

- variable dependency ?????? TODO　并行和多个相互影响很难处理
- Philosophical : Truth cannot be proved 真理不可证明。
- Mathematical : Fermat's theorem 费马大定理 方程$x^n+y^n = z^n, n \gt 2$ 没有整数解
- Physical : Three-body problem 三体问题
- Computational :  halting problem 停机问题
- Numerical :   initial value  sensitive problem 初值敏感问题

**7) Why a Euclidean-based measure would be most favored but usually impossible to obtain for a real world issue?**

(1)In real world, the features are heterogeneous and arbitrary. We need to find a way to measure them so that we can optimize the problem. 真实世界的特征往往是异质并且任意的，我们需要找到一种方法度量他们来优化问题。基于欧式空间的度量方法是最简单并且符合需求的。
(2)but the data in real world are heterogeneous, complex and dirty. Most of them are not independent and not measurable. We can use some methods to deal with the dimension space and apply feature selection to make them more measurable for Euclidean space but these methods are not general enough. 但是真实世界的数据是异质，复杂和脏的。大多数是相关的并且不可度量。我们有一些方法来处理维度空间以及进行特征选择来让数据在欧式空间内更可测。但是这些方法不够通用。

**8) What are some of the key requirements for a real issue to be formulated as a Euclidean problem?**

- Orthogonal, able to normalize, expand in the same way, the same measure to describe something (Better in a linear space) 正交，能够标准化，并且能以相同方式扩展,相同测度
- try difference loss to formulate Euclidean problem
- It is a dirty process for numerical problems.

**9) What would be the mathematical alternative frameworks to translate a non-Euclidean problem to mathematically appropriate solutions?**

Two major problem : select model, select feature.

(1) Recombine and select feature to make them independent. 

(2) Graphic model will be an alternative, describe features as a structure

**10) Why in general the complex and high-dimensional data (the so-called big data problem, n<<p) from the same “class” tend to have a low dimensional representation?**      

(1)Many features are redundant, not all feature are meaningful. Only some of the features are really determinant

(2)Human are limited that cannot easily perceive(感知) or deal with high-dimensional problem.

(3)Many features can cancel each other which makes some of them outstanding.

**11) Why we would prefer a low complexity model for a high complex problem?**

(1) There are two possibilities: under-fitting and over-fitting. The increase of any degree for model complexity will exponentially increase the risk of over-fitting

(2) Low complexity model tend to **capture the major features instead of some of the details**. The more details you try to explain, the more risk you’re going to be over-fitting, Thus we like major determinants(主要决定因素), principal components(主要组成部分) to increase **robustness**.

(3) Decrease the complexity and space, Ease the pain from feature selection and more easy to learn.

(4)Less possible to get a local optimal solution and singularity.

**12) What is a loss function? Give three examples (Square, Log, Hinge) and describe their shapes and behaviors.** ⭐️

(1) 定义: In machine learning, loss function is a function that measures the difference between the prediction result and the true result.损失函数是一个度量预测结果和真实结果之间差异的函数

(2) 三种损失函数

- Least Square: A quadratic surface $L(\beta) = C(Y-f_\beta(X))^2$
- Logistic: For logistic regression $L(\beta ) = log(1+e^{-y_if_{\beta}(x_i)})$
- Hinge: $max(0,1-y_if_\beta(x_i))$

(3) behaviors

- Least Square: It makes the assumption that the data distribution is Gaussian distribution and uses a homogeneous(同质) way to view data. 

- Logistic: More robust than Least Square. No assumption about
    data distribution. 

- Hinge: More robust. No universal assumption about distance...(YB's points of view)

**13) Using these losses to approach the linear boundary of a overlapping problem, inevitably (不可避免的) some risks will be incurred(产生); give two different approaches to remedy (弥补) the risk using the SVM-based hinge loss as an example.** ⭐️

- Use soft-margin
- Use $L_1$ or $L_2$ Norm to reduce dimension
- Use kernel method

**14) Describe biases (under-fitting) and variance (over-fitting) issue in learning, and how can we select and validate an appropriate model?** ⭐️

![微信图片_20190420110834](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420110834.png)

**15) How to control model complexity in the regression of a linear model? Are there supposed to be a unique low-dimensional model for a given high dimensional problem?**

(1) Using regularization term or Using L1 and L2 norm to reduce dimension

(2) yes TODO

**16) Using the Least Square as the objective function, we try to find the best set of parameters; what is the statistical justification for the Lease Square if the underlying distribution is Gaussian?** ⭐️

![微信图片_20190420111014](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111014.png)

![微信图片_20190420111017](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111017.png)

**17) Could you describe the convexity as to how it would facilitate(促进) a search? Using the Least Square-based regression and Likelihood-based estimation as the examples?** ⭐️

Convexity means the **global optimum is unique** and we can use Gradient-based method easily to find it. 

**18) Gradient Decent has a number of different implementations, including SMO, stochastic methods, as well as a more aggressive Newton method, what are some of the key issues when using any Gradient-based searching algorithm?**​ ⭐️

- The value of hyper-parameters like learning rate (step size).  超参数调参
- How to jump out of the local minimum. 如何跳出局部最小值
- The convexity of the problem.  问题的凸性
- parallel computation and speed up. 并行计算和加速

**19) What are the five key problems whenever we are talking about a learning process (Existence, Uniqueness, Convexity, Complexity, Generalizability)? Why are they so important? ​**⭐️

- Existence shows whether our model can converge.
- Uniqueness shows the difficulty of training. 
- If the problem is convex, we can solve it easily with Gradient-based searching algorithm and the global minimum always exists and is unique.
- Complexity shows the cost of training.
- Generalizability shows whether our model can achieve good result in new in test dataset and is robust in solving real world problem 

**20) Give a probabilistic interpretation(解释) for logistic regression, how is it related to the MLE-based generative methods from a Bayesian perspective?**​ ⭐️

TODO

![微信图片_20190420111505](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111505.png)

**21) What are the mathematical bases for the logics regression being the universal posterior for the data distributed in any kinds of exponential family members?**

 TODO

**22) Can you provide a probabilistic comparison for linear and logistic regression?**

 TODO

Logistic regression is to analyze the relationship between the probability of taking a certain value of the dependent variable and the independent variable, while linear regression is to directly analyze the relationship between the dependent variable and the independent variable.

- Linear: Regression From data itself overall,consistent,joint distribution
- Logistic: Classification From the distribution point of view likelihood-based, conditional distribution

**23) Why the log of odd would be something related to entropy and effective information?**

TODO

(1)log of odds:
$$
\begin{aligned}
logit(p)=&log\frac{p}{1-p} \\
\end{aligned}which is a sigmoid function.
$$
which is a sigmoid function. 

2)And the entropy has the form:
$$
\begin{aligned}
E[-log(p)] =& -\sum p_ilog(p_i) \\ 
=& -plog(p) - (1-p)log(1-p) \\
and\ derivative: \\
E'[-log(p)] =& -[log(p)+1]-[-log(1-p)-1] \\
=& -logit(p)
\end{aligned}
$$

(3)From a formal point of view, the two are also very related, so the log of odd is the direction of the entropy and effective information, 熵求导是log of odds

**24) Why often we want to convert a liner to a logistics regression, conceptually and computationally?**

(1) Conceptually: Liner talk about data and how data distributes from kind of truth, make your data consistent. while logistics talk about probability distributions for each of data, make them as different as possible 
(2) Computationally: Convert the problem to a binary problem. Limit the result to 0-1. The computation is less complex. Efficient because does not need to calculate the exact number of x. More robust to the noises

**25) Compare the generative and discriminative methods from a Bayesian point of view?** ⭐️

(1) 生成模型试图对联合概率建模，而判别模型试图对条件概率建模

(2) 通常来说，生成模型比判别模型需要更多训练样本，因此有高计算复杂性的问题

(3) 但是生成模型会提供更多关于数据如何生成的内在信息

(4) 判别模型可能有概率解释或者没有，而生成模型完全基于概率理论

![微信图片_20190420111654](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111654.png)

**26) What are the most important assumption for something Naïve but still very effective? For instance for classifying different documents?**

TODO

Joint Distribution

**27) What would be the most effective way to obtain a really universal prior? And what would be the most intriguing implications for human intelligence?** 

TODO

Knowledge Graph

**28) What are the key advantages of linear models? But why linear model tends not expressive?** ⭐️

(1) Key advantages of linear model: Such framework minimizes interactions between different factors, and also has very low computational complexity. Integration(积分) of linear models can solve complex problems. 减小了不同特征的关联性，同时具有低计算复杂度
(2) Not expressive: Linear regression is poor when the variables are nonlinear.And it is not flexible enough to capture more complex patterns, making it difficult and time consuming (耗时) to add correct interaction terms or use polynomials. 变量是非线性是表现很差，不具有抓住复杂模式的能力，增加正确的交互项或者使用多项式十分困难和耗时.

![微信图片_20190420111831](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111831.png)

**29)  What are the key problems with the complex Neural Network with complex integrations of non-linear model? **

(1)It sustains the **curse of combinatorial explosion** such as network topology and a dramatic huge number of parameters. 具有组合爆炸的诅咒，例如网络拓扑和非常多的参数

(2)Multiplication decreases the degree of **convexity** and therefore the model becomes more **sensitive to the initial value**. 乘法会降低凸度，因此模型对初值更敏感

感觉28,29的重点在interaction上，好处interaction多，坏处interaction少

**30) What are three alternatives to approach a constrained maximization problem?**⭐️

1. Solving its dual problem (Lagrange Multiplier拉格朗日乘子) 解决对偶问题
2. Find its equivalent problems (modify objective function 修改目标函数)
3. Using kernel tricks 使用核技巧 

**31) What is the dual problem? What is strong duality?**⭐️

对偶问题就是min(max) 与max(min)的转化

强对偶性是指对偶问题的最优解相同

![微信图片_20190420111954](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111954.png)

![微信图片_20190420111958](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420111958.png)

**32) What are the KKT conditions? What is the key implication of them? Including the origin of SVM?**⭐️

(1) KKT condition：

Consider an optimization problem:
$$
\begin{aligned}
\min_{x}&f(x) \\
s.t.& g_i(x)\le 0,i=1,...,k \\
&h_j(x)=0,j=1,...,l
\end{aligned}
$$
It's Lagrange  function is defined as
$$
L(w,\alpha,\beta) = f(x)+\sum_{i=1}^{k}\alpha_ig_i(x)+\sum_{j=1}^{l}\beta_{j}h_j(x)
$$
KKT conditions: **TODO**

=======================================

![微信图片_20190420112145](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112145.png)

![微信图片_20190420112149](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112149.png)

![微信图片_20190420112300](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112300.png)

TODO SVM形式有问题 以及最终解法推导

**33) What is the idea of soft margin SVM, how it is a nice example of regularization?**⭐️

![微信图片_20190420112335](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112335.png)

**34) What is the idea of kernel? Why not much additional computational complexity?** ⭐️

(1)整个计算过程中,x只出现在内积计算过程$x_i^Tx_i$中，因此我们可以引入一个核函数$k(x_i,x_j)$来代替原来的$<x_i^T,x_i>$，这个通常被称为kernel trick

(2) 本质上来说，核函数对应着一个映射特征空间，因为我们只是进行的了替换，而不知先投影在计算内积，所以没有正价额外的计算复杂度。

(3) 因为核函数 $K(x,y)= \phi(x)^T\phi(y)$，kernel trick实际上合并了一些项来减少运算，例如高斯核将计算时间从无限降低到了有限时间

![微信图片_20190420112425](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112425.png)

**35) What is the general idea behind the kernel? What key computation do we perform? Why is it so general in data modeling?**​ ⭐️

![微信图片_20190420112505](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112505.png)

**36) Why we often want to project a distance “measure” to a different space?**  ⭐️

In many situations, the real data space is usually non-Euclidean. Therefore, we want to project a distance measured in Euclidean space to its real space. (nonlinear $\Rightarrow$ linear) 

在很多情况下，实际数据空间通常是非欧式空间，因此我们想将欧式空间的距离测度投射到实际数据空间。完成非线性到线性的转换

**37) What a Turin machine can do? What some of the key computable problems a Turin machine can do?**

TODO

**38) What a Turin machine cannot do? What some of the key computable problems a Turin machine cannot do?**

TODO 不存在真正的并行，图灵机不可能实现并行

**39) Give your own perspectives on the Hilbert No.10 problem in the context of computational limit.**

Hilbert No.10 problem: Is there a general algorithm which, for any given Diophantine equation(不定方程), can decide whether the equation has a solution for all unknowns with integer values.对于任意多个未知数的整系数不定方程，是否存在一个可行的算法，使得借助于它，通过有限次运算，可以判定该方程有无整数解。

(whether a algorithm can lead to a solution, the limit of Turin machine, math, philosophy, answer is no) 不存在，这个问题指出了，存在一些问题，我们无法通过有限次运算解出。指出了图灵机计算的极限。

**40) Give your own perspectives on the Hilbert No.13 problem in the context of mathematical limit.**

TODO

Hilbert No.13 problem: Can a given high-dimensional problem be described by a composition with a finite number of bivariate functions(二元函数)?

( motivation for anything.  example: RNN support function)

**41) Discuss human intelligence vs. Turin machine as to whether they are mutually complementary, exclusive, or overlapping, or contained into each other one way or another.**  补充，排斥，重叠，包含

人类能够高效的利用先验知识，但是不能高效准确的进行重复计算

而图灵机恰恰相反，我认为两者是mutually complementary。

**42) Explain Bayesian from a recursive point of view, to explain the evolution of human intelligence.** 

Recursion to update the prior and structure, search problem 

instinct(本能) => passive => active => explore => inference(推理)

**43) What are computational evolution basis of instinct, attention, inspiration, and imagination?**

all is priors, particularly in the context of Bayesian, can be shared

**44) Explain the core idea of machine learning to decompose a complex problem into an integration of individual binary problems, its mathematical and computational frame works.**  

Hilbert 10,13 problems. things are logic, we can convert complex problems into combination  of binary problems by using universal frameworks or math frameworks. The structural and logical recombination of binary processes

**45) What are the limitation of Euclidean (Newtonian) basis, from the space, dimension, measure point of view?**

The situation is too ideal,(homogeneous, orthogonal, normalized)齐次，正交，标准

In reality, it's very complex, 高维导数过于复杂，不同的特征往往测度不同，特征往往是混杂的并且相互依赖。 Multiple differential(depends on each other) three body problem

**46) Why differentials of composite(组合) non-linear problems can be very complex and even singular?**  

函数复杂了难以求导，更难以用梯度去逼近。此外系统复杂，很容易对初值敏感

**47) What is the basis of Turin halting problem? And why temporal resolution (时间分辨率concurrency并发) is a key for logics and parallelism?** 

(1) 没有真正的并行 no true parallelism. we need a order to deal with the data

(2) if temporal resolution is good enough, then we can achieve parallelism with some error

**48) What is mathematical heterogeneity(异质性) and multiple scale(多尺度)?** 

反义词：heterogeneous<=>homogeneous; multiple scale<=>local/own
mathematical heterogeneous: we need to integrate independent problems.
Multiple scale: Combine all the local scales and normalize

数学异构：我们需要整合独立的问题。
多尺度：组合所有局部尺度并标准化

**49) Explain convexity of composite functions? How to reduce local solutions at least in part?**

(1)Composite is generally not convex but sometimes can be convex

(2) 

a) fix one and only change one
b) divide it into partitions/regions which are convex.

**50) Why local solution is the key difficulty in data mining, for more generalizable learning?** 

Because there are many alternative way to relate X and Y .关系太多很难泛化

### Probabilistic graphical model

**1) Compare the graphical representation with feature vector-based and kernel-based representations;**⭐️

图形：表示和可视化变量之间关系的直观方式（条件独立性），也可以表示相互依赖性

基于矢量的，基于内核的：关于数据分布的统计视图（坐标变换，方差）独立于数据

基于矢量：降维。 

基于内核：投影，非线性和坐标转换

![微信图片_20190420112632](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112632.png)

**2) Explain why sometime a marginal distribution边缘分布 has to be computed in a graphical model;** ⭐️

![微信图片_20190420112834](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420112834.png)

**3) Why class labels might be the key factor to determine if presumptively different data distributions can be indeed discriminated?** 

Assume labels are outstanding for each class:
From a Bayesian point of view: label equals to constrain which is ideal condition or truth and it should be as universal as possible. 
If label is ambiguous, that means it will vary from each individual. This will make it difficult to learn.

标签等于约束哪个是理想的条件或真理，它应该尽可能地普遍。
如果标签含糊不清，这意味着它会因每个人而异。 这将使学习变得困难

**4) Why knowledge-based ontology (representation) be a possible solution for many prior-based inference problems?**  

$P(x,y)= P(y|x)P(x)$

A knowledge-based ontology is a universal ontology which is from the understanding point of view. By using the prior structure, a distribution of data can be guaranteed.

**5) Why a graphical model with latent variables can be a much harder problem?**⭐️

- 有了隐藏变量，通常数据分布会更加复杂。同时除了寻找模型参数还要求出隐藏变量。
- 大多数问题数据可以被EM算法解决，但是EM算法不保证全局解
- 隐藏变量是通过期望获得的，这只是一个近似的方法
- 这个问题是非凸的

![微信图片_20190420113044](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113044.png)

**6) What is the key assumption for graphical model? Using HMM as an example, how much computational complexity has been reduced because of this assumption?**⭐️

条件独立和马尔科夫特性

马尔科夫过程，假设当前状态只依赖于上一个状态，这个可以简化计算过程，不需要太多乘法

![微信图片_20190420113125](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113125.png)

**7) Why does EM not guarantee a global solution? What is a simple proof for that?**⭐️

![微信图片_20190420113206](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113206.png)

$P(X,H;\theta)$可能是非凸的，

同时Jensen不等式的条件要求f(x)是凸的，但是f(x)= log(g(x))不一定是凸的

**8) Why is K-mean only an approximate and local solution for clustering?**⭐️

![微信图片_20190420113331](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113331.png)

K-Means 是EM的一个特例

**9) How to interpret the HMM-based inference problem from a Bayesian perspective, using the forward/backward algorithm?**⭐️

![微信图片_20190420113417](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113417.png)

**10) Show how to estimate a given hidden state for a given series of observations using the alpha and beta factors;**  ⭐️

![微信图片_20190420113506](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113506.png)

**11) How a faster inference process would be constructed, given a converging network?**

Let the priors be consistent and big. By applying multiplication, all the priors must be as big as possible
to faster the inference(Otherwise will cancel others).
From structural point of view, we want to use more converging products，改变图的结构，数据的顺序好，深度变浅，上游的数据重要

**12) How can an important node (where inference can be significantly and sensitively affected) be detected using an alpha and a beta process?**

By calculating $\alpha \beta$, where the value changes dramatically is the import node

**13) Why often an alpha process (forward) is more important than beta (backward)?**

Forward process is to update the prior which is more important for human.前向传递是在更新先验

**14) What are the key differences between an alpha and a beta process from human and machine intelligence point of views?**

Human: take advantage of priors, do not need whole sequence. 人类可以利用先验知识
Machine: Have to exhaust all the possible results. Can only use data to maximize, to regression.

**15) How data would contribute to the resolution of an inference process from a structural point of view?**

A better likelihood, 贝叶斯公式

**16) For a Gaussian graphical model, what is the implication of sparsity for such a graphical model? How is such sparsity achieved computationally?**⭐️

(1)Most edges have zero weights. i.e. sparse adjacent matrix. (Pruning)

(2)how

- Applying SVD in dimension reduction. Simplify the computation.
- using L1 norm
- use few features as possible 

**17) Explain the objective function of Gaussian graphical model? Thus the meaning of MLE?**

use 

**18) How a row or column-based norm (L1 or L2) can be used to construct hub-based models? And why this might be highly applicable?**

by row or column L1 L2 norm , it become sparse in row and column, this mean there many zeros in row and column, so we have many central point thus hub-based. 

since there many zeros (sparse),so it's easy for computing ,thus highly applicable.

不仅全局稀疏，而且集中在某行某列，图中有中心节点。

**19) How Gaussian graphical model be used to model a temporally progressive problem? Why still a simple 2d matrix comprasion problem can still be very demanding computationally?**  

将过程看成高斯图的变化，但是高斯图之间的差异可能很大，因此我们要使用上一时刻的约束俩约束现在的时刻，以使模型缓慢变化，能够描述一个过程。

2D 矩阵的比较是图的同构问题，比较两个图是否同构是及其困难的，需要极高的计算量。

**20) Why a spectral or nuclear norm would be better than Frobenius norm to obtain a sparse matrix?**  

Frobenius norm 只使用了部分数据，并没有关注数据之间额关系

nuclear norm 使用点积等形式，关注了不同数据的关系

### Dimension reduction and feature representation

**1) PCA is an example of dimensional reduction method; give a full derivation of PCA with respect to its eigenvectors; explain SVD and how it is used to solve PCA;**⭐️

![微信图片_20190420113803](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113803.png)

Answer from Newly:

- Interpretation of PCA: 

![微信图片_20190420163253](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420163253.png)

- Interpretation of SVD:

![微信图片_20190420163353](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420163353.png)

![微信图片_20190420163452](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420163452.png)

![微信图片_20190420164654](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420164654.png)

![微信图片_20190420164658](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420164658.png)

![微信图片_20190420164703](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420164703.png)

**2) Compare regular PCA with the low-ranked PCA, what would be advantage using the low-ranked PCA and how it is formulated?**⭐️

TODO

![微信图片_20190420113847](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420113847.png)

Extend reading: SPCA can restrict the process of linear combination when calculate the eigenvectors. So that the results have a stronger real-world meaning and easier to be explained.
Sparse PCA: https://blog.csdn.net/zhoudi2010/article/details/53489319 

![微信图片_20190420121816](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420121816.png)

SPCA原始文献：[H. Zou (2006) Sparse principal component analysis](http://www.tandfonline.com/doi/abs/10.1198/106186006X113430) 

**3) What is the difference between a singular value and its Eigen value? Explain the resulting singular values of a SVD for how the features were originally distributed;** 006:18:27

Answer from Newly:

![微信图片_20190420171110](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420171110.png)

![微信图片_20190420171114](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420171114.png)

Question2:

Two scenarios for how the features might be distributed:

- previous feature independent from each other. Original features are informative. PCA is not very effective. 特征之间相互独立，原始特征信息量很大，PCA效率低
- features with a lot of redundancy. PCA has to be effective. 特征冗余多，PCA十分有效

**4) What is the key motivation (and contribution) behind deep learning, in terms of data representation?** ⭐️

Deep learning is a kind of data representation learning. It automatically discovers the representations needed for feature detection or classification from raw data. This replaces manual feature engineering and allows a machine to both learn the features and use them to perform a specific task. And DL does well in complex representation.

Split and combination. Learn complex, abstract feature combination. 

![微信图片_20190420121936](./Images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190420121936.png)

- Motivation: 构建很多隐藏层的模型来学习如何组合数据，学习解释数据背后的理论结构和逻辑
  1. De-compositional approach: multiple building blocks that learns how original data recombine together.
     2. interpret underlying mechanism, underlying structure, and underlying logics
  2. universal approximation theorem
- Contribution:  能够自动的从原始数据中找到需要的特征检测和表示方法，通过逐层的特征变换，能够表示出复杂的数据特征
  - automatically discovers the representations needed for feature detection or classification from raw data
  - complex representation for the structure of the data

**5) Compare the advantage and disadvantage of using either sigmoid or ReLu as an activation function?**

Loss: 

- Sigmoid:
    -  output probabilistic results. But it has gradient vanishing problems.梯度消失问题 
    - 非线性，无梯度爆炸问题
- ReLU: 
  - 有梯度爆炸(blow up activation)和死亡( dying ReLU)问题
  - avoid gradient vanishing problems. Computing gradient is faster.  避免梯度消失，计算快
  - $x$ becomes arbitrary. Use $b$ (斜率还是截距) to calibrate each activation function. 
  - Combination of ReLU functions is less expressive, because the ReLU function is simple and of low-order. 表达能力变差了，因为ReLU是简单和低次的

**6) Discuss matrix decomposition as a strategy to solve a complex high dimensional problem into a hierarchy of lower dimensional combinations?**

TODO========================================

=============================================================

Mathematical justification for deep learning.

(YB writes something on the blackboard...which explains this problem. Something about $n\log n$ overhead for $n^2$ capacity)

Pruning/Dimension Reduction: Sparsity.

复杂度低，能够控制矩阵的稀疏度，通过pooling(降低连接性和冗余)

**7) Discuss convexity of composite functions, what mathematical strategies might be used to at least reduce local solutions?**     

DL: vertically product of composite functions and horizontally summation of them. This leads to many local solutions.

Strategies to reduce local solutions:

- ReLU: simplify combination.
- Partition neural networks 

**8) Why normally we use L2 for the input layers and L1 for the actual modeling? Explain why still sigmoid activations are still used for the output layers?** 

Input layers of L2 and hidden layers of L1:

- L2: smooth inputs. Because some data/ features still are really big. L2 cancel to relative weight and average inputs (not to different from each other) 平衡输入，一些feature可能过大
- L1: for sparsity. Keep pruning connections between the layers. Sparse networks are computationally tolerant. 稀疏度，减少连接，提高计算容忍度
- Sigmoid (softmax): re-normalize and recalibrate(校准) the outputs. Get probabilistic outputs (Logistic Loss) and try to evaluate from probabilistic distribution distance or entropy point of view. Sometimes we use $tanh()$  从概率分布和熵的角度来评价输出

**9) What would be the true features of an object modeling problem? Give two examples to highlight the importance of selecting appropriate dimensions for feature representations;**⭐️

================TODO

truth table: only 3 dimensions are critical other boolean variables are useless. (dimension reduction and feature selection) use 3D instead 8 bits one-hot

Posture of human face: high-dimension image is good ---- two/ three-dimension representation is good.3D重构比较好

============

Have strong information, low-rank
DL can dig out the underlying relationship within the data and get the "true" feature
Problems: Basis function selection, ways of combination (topology), BP(gradient vanish and explode), memory (RNN, LSTM), Computational complexity, Over/Under-fitting 

head location: reconstruct 2D pic to 3D space.
truth table: only 3 dimensions are critical other boolean variables are useless. (dimension reduction and feature selection) use 3D instead 8 bits one-hot

Posture of human face: high-dimension image is good ---- two/ three-dimension representation is good.3D重构比较好

YB's own answer: Two examples

- Posture of human face: high-dimension image and two/ three-dimension representation is good.
- Simulate Boolean algorithm: how many sigmoid functions you need? Three gates that construct 8 bits truth tables. It is three-dimensional (I don't know what this is about)

**10) Why does the feature decomposition in deep learning then a topological recombination could make a better sampling? What would be the potential problems making deep learning not a viable approach?** 006:40:53

DL is able to decompose data into different primitive parts. These parts cam recombine those other parts. This exponentially or factorially increase the data for possible combinations. This increase the sample capacity and can generate sample $\Rightarrow$ GAN

Potential problems: (on slides)

- Local solutions
- Vanishing gradients
- ...

**11) Explain the importance of appropriate feature selection being compatible with model selection in the context of model complexity;**⭐️ 006:44:56

The feature should be compatible with the representation ability of the model. Otherwise, over-fitting or under-fitting will be likely to happen.需要和模型表达能力匹配，不然容易过拟合或者欠拟合

Higher dimensional data are more likely to over-fitting. 

The feature selection should match the model selection. 

**12) What would be the ultimate and best representation for a high dimensional and complex problem? How this might be possibly achieved?** 006:46:14

Everything should be modular and re-useful. Everything should be combination of building blocks. Problems should be reorganized by the combination of the structural, logical, and conceptual parts instead of case by case. (Key word: modular combination)模块化组装，结构化，逻辑化和概念化的组装

**13) How RNN can be expanded our learning to fully taking advantage of Turing machine? What RNN can whereas CNN cannot do?** 006:47:38

RNN & CNN:

RNNs are more capable of achieving something beyond simulation of functions. RNN begins to simulate some logical process (memorizing, decision, and stopping). RNN learns not only a process function but also Turing process where logics can be simulated (recursions...). CNN cannot do this.  考虑了逻辑过程，记忆决策和停止，而不是函数

From Turing point of view: CNN just changes the input and get different output. CNN does not consider how to connect them in a logical, recursive, parallel, and interactive process (No logic and stops for input and output relationship). RNN expands that (Theorem). 

只能须爱惜函数，不考虑从逻辑，循环和并行以及交互过程

**14) What is the central additional difficulty of RNN compared to CNN?**  006:50:24

Difficulty: RNN异质多尺度系统，有local loss. 怎么去结合局部loss, 没有全局loss

- RNN is a heterogenous and multiscale system. RNN blocks are more individual, and we need to combine them together. RNN itself has local loss. The problem is how to combine local loss (linearly/weighted). there is no global loss. 

**15) In the activation function, there is a constant term “b” to learn, why it is important?**

(Sorry. The audio does not contain this part...)

**16) LSTM integrate short and long term processes, what is the central issue to address to achieve at least some success?** 007:00:09

- Local & global loss

**17) The difference between value-based vs. policy-based gradients?** 

In reinforcement learning: value-based v.s. policy-based （not covered  in test)

**18) Why dynamical programming might not be a good approach to select an optimal strategy?**

（not covered in test)

**19) Explain an expectation-based objective function?**

（not covered  in test)

**20) Explain Haykin’s universal approximation theorem?**

（not covered  in test)

### General problems

**1) In learning, from the two key aspects, data and model, respectively, what are the key issues we normally consider in order to obtain a better model?** ⭐️

- model selection
- feature selection
- model/ feature compatible 兼容
- dimensionality reduction 姜维
- model robustness  模型鲁棒性
- sample complexity 采样复杂度
- model complexity 模型复杂度

**2) Describe from the classification, to clustering, to HMM, to more complex graphical modeling, what we are trying to do for a more expressive model?**⭐️ 007:03:43

Humans solve complex problems using priori, while machines do so using combinations of basic functions
Using model combination to get a more complex and expressive model, and also fit more complex problem. 通过模型的组合

From less expressive to more expressive (increase of model expressiveness): 表达性在提高

- For classification: we use $\{+1, -1\}$ or $ \{0, 1\}$
- Cluster: we look at $\{0, 1\}$ in terms of their combination $\Rightarrow$ Marginal
- HMM: model process as a inference and condition that can be updated

**3) What are the potential risks we could take when trying to perform a logistic regression for classification using a sparsity-based regularization?**⭐️ 007:06:14

L0: NP-Complete problem
L1: 

- overlooking certain parameters due to different order of data. (sequential risk) Because we need to pick which dimension to maintain instead of by itself. L1 Norm could somehow throw away important dimensions. 
- we also risk of underfitting

For linear model:

- sequence risk，可能会因为输入顺序不同，而忽略了某些特征。没准会扔掉重要特征
- measures is not consistent 测度可能不一致
- model risk: λ can be either too great or too small, causing under fitting or over fitting problems.  罚项系数可能过大或过小，导致under fitting或者 over fitting

**4) Give five different structural constrains for optimization with their corresponding scalars;**⭐️ 007:08:45

- L1 norm: $||x||_{L1} = \sum_{r=1}^{n} |x_i|$ 

  more aggressive

- L2 norm: $||x||_{L2} = \sqrt{\sum_{r=1}^{n} x_i^2}$

  more smooth (想象两者的图像)

Matrix Norm
$$
||A||_p = \left(\sum_{i}^{\min\{m,n\}} \sigma_i^p(A) \right)^{1/p}
$$

- Frobenius norm ($p = 2$): $||A||_{F} = \sqrt{\sum_i^{m}\sum_j^n a_{ij}^2}$  弗罗宾尼斯范数 平方和
- Nuclear norm ($p=1$): the sum of singular value 核范数 所有值之和
- Spectrum norm ($p = \infty$): maximum singular value 谱范数 最大值

**5) Give all universal, engineering, and computational principles that we have learned in this course to obtain both conceptually low-complexity model and computationally tractable algorithms?**

Locality, gradient, linearity, convex, low-rank, combination, binary, priori (Bayes), Markov, expectation,
recursion, measure 

YB's audio answer:

- Locality 局部性
- Convexity 凸性
- Linearity 线性
- Sparsity: reduce complexity 稀疏性，降低复杂度
- Low-rank representation: reduce dimension 低秩表征，降维
- Prior (Bayes) 先验
- Markov 
- Entropy
- Gaussian

**6) Why data representation is at least equally as important as the actual modeling, the so-called representation learning?** (YB skipped this)

model selection and feature selection are closely associated with one another;

 data representation needs to be compatible with the model and capture necessary features learning the combination of features and the relationships between features

Note: machine learning tasks such as classification often require input that is mathematically and computationally convenient to process

**7) How does the multiple-layer structure (deep learning) become attractive again?** (YB skipped this)

People realized that structure cannot be imposed on models, so deep learning first learns the structure of the data and finds the relationships. With more layers, the representation of features becomes richer (?)

Deep learning resolves nonlinear thing with multiple linear combination.

Note: the increase in computational resources and the utilization of GPU acceleration, big data

**8) Discuss Turin Completeness and the limit of data mining;** 

Think for the rest of your life. 

Limits:

- Singular issue
- Multiscale
- Local solution

Leads data mining to be empirical and heuristic

**9) Discuss general difficulties of using gradient for composite functions or processes;** (YB skipped this)

**10) What is the trend of machine learning for the next 5-10 years?** (YB skipped this)

deep learning theory (???)
parallel (???)
reinforcement learning (???) 

### Previous Exam

In one problem: one part is substantial and one part is general

**1)     SVM is a linear classifier with a number of possible risks to be incurred, particularly with very high dimensional and overlapping problems. Use a simple and formal mathematics to show and justify (a) how a margin-based liner classifier like SVM can be even more robust than Logistic regression? (b) how to control the overlapping boundary?** 

- Write loss function for SVM, soft margin, lagrangian

Question 1:

- Dual approach: SVM use the inner product to measure similarity

Question2:

- Soft margin: Formulation. More robust.
- Kernel method: Formulation. Nonlinear separation.

**2)     Why a convolution-based deep learning might be a good alternative to address the dilemma of being more selective towards the features of an object, while remaining invariant toward anything else irrelevant to the aspect of interests? Why a linear regression with regulations would result in features which are usually conceptually and structurally not meaningful?**  

Question 1:

- Convolution averages the data. CNN focus not only on the features but also the background. 
- Convolution measures the pattern of the data. (context) 

Question 2:

- Features are combinations of patterns (context). Pattern/ context v.s. the individual feature

**3)     There are a number of nonlinear approaches to learn complex and high dimensional problems, including kernel and neural networks. (a) please discuss the key differences in feature selection between these two alternatives, and their suitability; (b) what are the major difficulties using a complex neural network as a non-linear classifier?** 007:28:18

Question 1:

- DL decompose them into building blocks and learn how to construct them. 
- DL understand the abstraction of the problem and the context.
- DL's risk is mathematical: local solution...
- Kernel emphasize the training and testing of the data. (data driven approach)

Question 2: 

- Composite function & its differentials
- local solution
- gradient vanishing
- singularity

Both problems are exhaustive (inverse and ill-posed).

**4)     For any learning problems, (a) why a gradient-based search is much more favorable than other types of searches? (b) what would be the possible ramifications of having to impose some kinds of sequentiality in both providing data and observing results?** 007:32:44

Question 1: 



Question 2: 

Gradient is a sequential process. It takes multiple sequential process for a parameter to be learned or to be searched. It depends on the sequentiality of how data is provided. The two process has to be impartible. You cannot have model learned but the data comes later.

**5)     Please use linear regression as the example to explain why L1 is more aggressive when trying to obtain sparser solutions compared to L2? Under what conditions L1 might be a good approximation of the truth, which is L0?**   

Question 1:

very simple.

Question 2:

It is a sequentiality issue. The model being reduced has to be supported by the data. Your data has to be in pace with the dimension reduction. L0 means you have to do exhaustive search and look for the best possible combination. In the actual regression, it is a sequential process. It depends on timing of this issue. If your data is mixed, the timing is not important. 

**6)     What is the key difference between a supervised vs. unsupervised learnings (where we do not have any ideas about the labels of our data)? Why unsupervised learning does not guaranty a global solution? (use mathematical formulas to discuss).**       

Question 1

YB writes something on the board. 

Differences: 

- Global solution v.s. Local solution

Question 2

Discuss about EM algorithm (Mathematical Formulation).

**7)     For HMM, (a) please provide a Bayesian perspective about the forwarding message to enhance an inference (using a mathematical form to discuss); how to design a more generalizable HMM which can still converge efficiently?**

Question 1:

Write down the forward process of HMM

Discuss the prior update. 

( Relate $\alpha$ to human and $\beta$ to machine)

Question 2:

We want this eventually converges but not too subtle. Progression of $\alpha$ is sudden or slow during which a number of possibility has to be involved. Therefore, more possible scenarios have to be included. That is more generalizable. 

**8)     Using a more general graphical model to discuss (a) the depth of a developing prior-distribution as to its contribution for a possible inference; (b) how local likelihoods can be used as the inductions to facilitate the developing inference?**

Question 1:

How long or how deep the alpha has to be propagate. Sometimes we need to the alpha to change as quick as possible and sometimes we want to get as deep as possible. We take multiple additional factors into account. The decision is more comprehensive. If you go too quick in the beginning, you can actually be biased by some local factor. 

Question 2:

Posterior = likelihood x prior. This update the prior by times the local likelihood.   

**9)     Learning from observation is an ill-posed problem, however we still work on it and even try to obtain convex, linear, and possibly generalizable solutions. Please discuss what key strategies in data mining we have developed that might have remedied the ill-posed nature at least in part? Why in general linear models are more robust than other more complex ones?**

Question 1:

YB skipped this problem.

Question 2:

YB skipped this problem.

**10)   Using logistic regression and likelihood estimation for learning a mixture model (such as the Gaussian Mixture Model), please using Bayesian perspective to discuss the differences and consistencies of the two approaches; why logistic function is a universal posterior for many mixture models?**

Question 1:

First one is more sensitive towards the boundary data. Second is a generative model that is more robust which takes more data into account. Regression has no assumption about the model. The two approach is unnecessarily identical

Question 2:

As long as a distribution is exponential, logistic function is a universal posterior for it. 

