# Tensor Programs V

## Abstract
Hyperparameter turning is an expensive process especially at scale. This paper shows that at Maximal Update Parametrization $\mu P$, optimal values for certain hyperparameters remain stable even with model shape changes. This insight leads the authors to design a new paradigm of hyperparameter tuning: parametrize the target model in $mu P$, tune the hyperparameters indirectly on a smaller model, and zero-shot transfer to them to the full-sized model. The authors claim to have applied this method on a 13M parameter model to outperform BERT-large which is 350M. They also transferred from 40M to outperform the 6.7B GPT3 model.

## Introduction
**Algorithm 1: Tuning a Large Target Model via $\mu$Transfer**
1. Parametrize target model in Maximal Update Parametrization ($\mu P$)
2. Tune a smaller version (in width and/or depth) of target model
3. Copy tuned hyperparameters to target model

This method seems obviously promising. It has a few limitations, however. Namely, it cannot transfer regularization hyperparameters, so it's generally not applicable for finetuning pretrained models.

## Parametrization matters: a primer
This section gives a primer on why the correct parametrization can allow hyperparameter transfer across width

The Cental Limit Theorem decrees that if a dataset $x_1, ..., x_n$ is iid with mean $\mu$ and standard deviation $\sigma$, the distribution of the sample's means $\frac{1}{\sqrt{n}}(x_1+...+x_n)$ converges to a standard Gaussian distribution as $n \rightarrow \infty$.

If the scaling factor on that previous expression is changed to something like $1/n$, then $c_n(x_1+...+x_n) \rightarrow 0$. Or, if the scaling factor is 1, then $c_n(x_1 + ... + x_n)$ blows up in variance as $n \rightarrow \infty$

Now, suppose we're trying to minimize the function
$$F_n(c) = \mathbb{E}_{x_1,...,x_n}f(c(x_1+...+x_n))$$

over $c \in \mathbb{R}$. If we reparametrize $c = \alpha / \sqrt{n}$, then by CLT, the function stabilizes into a function of $\alpha$ as $n \rightarrow \infty$. By this token, for a sufficiently large $n$, the optimal $\alpha^*_n$ should be close to $\alpha^*_N$ for any $N > n$. This is a mathematical backing for the idea of *transfering* the optimal $c^*_n$ for a smaller problem to a larger one.

In this scenario, the data distribution $x_1,...,x_n$ is akin to the randomly initalized parameters of a width-$n$ neural network. $c$ is akin to a hyperparameter such as learning rate, and $f$ is the test-set performance of this network post-training such that $F_n$ gives its expectation over ransdom initalizations. As was illustrated, if we parametrize the LR and other hyperparameters correctly, we can directly copy the optimal HPs for a narrower network into a wider one.

Note that to ensure transferability of any hyperparameter, it's not sufficient to reparametrize only the one in question. We need to identify and reparametrize all the hyperparameters.

## Hyperparameters don't transfer conventionally
To illustrate why standard parametrization is wrong, the authors examined the shift in hyperparameter optima on a simple 2 layer MLP. The optimal learning rate shifted by an order of magnitude as the width increases from 256-8192. This observation held true for more complex architectures such as Transformers.

## Unlocking zero-shot hyperparameter transfer with $\mu P$
### MLP with $\mu P$
To switch the MLP from the previous example to $\mu P$, the initialization of the last layer, the learning rates of the first and last layer, and the biases need to be modified.
$$\text{initialize}\ W^1 \sim N(0, \frac{1}{d_{in}}), W^2 \sim N(0, \frac{1}{n}), W^3 \sim N(0, \frac{1}{n^2}), b^{\{1, 2\}}=0 \\ \text{with SGD learning rates}\ \eta_{w^1} = \eta_{b^1} = \eta_{b^2} = \eta n, \eta_{W^2} = \eta, \eta_{W^3} = \eta n^{-1}$$

Here, $\eta$ specifies the "master" learning rate. The basic form makes clear the scaling with width $n$ of the parametrization, but in practice a multiplicative constant is inserted in front of each appearance of $n$.

# Experiments
Their empirical method is as follows. They train a 2 layer (2 self-attention blocks) pre-layernorm $\mu P$ Transformer with 4 attention heads. They then sweep one of four hyperparameters (learning rate, output weight multiplier, initialization standard deviation, and learning rate schedule) while fixing the others and sweeping along width and depth. 

## Transformer on IWSLT14 De-En
The model used here is the default IWSLT post-layernorm Transformer with 40M parameters. For $\mu Transfer$, the hyperparameters were tuned on an architecture a quarter of the width, amounting to 4M parameters. For this experiment, the learning rate, output layer parameter multiplier, and attention key-projection weight multiplier was tuned via random search.

## Implementing $\mu Transfer$
We can enable $\mu Transfer$ by simply reparametrizing the desired model in Maximal Update Parametrization ($\mu P$). While conceptually simple, swtiching from SP to $\mu P$ can be error-prone as deep learning frameworks such as Pytorch are built around SP. So, the authors have built a tool that aims to minimize code changes when switching to $\mu P$, and keep model behaviour invariant at a given base model shape. Thus, the [mup](https://github.com/microsoft/mup) package is built