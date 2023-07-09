# Overview

Scaling laws is, as the name implies, the study of how a model behaves as it's scaled up. The past decade of deep learning research has shown that executed correctly, scaling laws can pose an even greater opportunity for improving our architectures than the pursuit of better inductive biases.

Practically speaking, this is done by exploring scaling strategies. These strategies serve as paradigms for how one might scale up a neural network, for example studying how increasing depth, width, and data improves performance. Functional forms of said strategies are composed, allowing the researcher to accurately predict how a model will behave under a certain paradigm given a fixed computational budget.

Finally, having observed several scaling strategies, a skilled researcher can determine the *best* way to scale a model up such that it performs best under a training regime. He can also start to piece together *why* this strategy is the best, and work to build architectural modifications to further improve performance.

## Changelog
### 2023/07/10
- [[Achieve stability]]