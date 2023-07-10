# Achieve stability

As of now, a training regime has been finalized. Learning rate and weight decay sweeps have been conducted, setting a good baseline for future scaling strategies.

![W&B results towards final steps of training regime](2023-07-09-13-06-43.png)
However, upon observing the depth scaling strategy, results are ambiguous. There's no clear best model, and worse yet, performance does not follow any clear trend (the expected one of course being an increase in performance as depth is scaled). 

This is a massive issue because the core feature of scaling laws research is the predictive power of the functional form. If every epoch regime yields a different functional form (even regimes within 1 epoch of another), the predictive power is essentially zero. We're looking for <ins>stable</ins> learning curves with clear monotonic performance trends. Larger models should be consistently more performant than smaller ones, and the observed forms shouldn't change much for epoch regimes within reasonable bounds.

## Diagnosis

>Start out with the tiniest provable experiment when debugging. Incrementally add complexity.

![Depth 9 vs depth 5](2023-07-09-15-43-52.png)

What changes between depth 5 and depth 9?

- Depth 9 starts out much worse than depth 5
- Depth 9 crosses depth 5 at the 15k'th step
- Depth 9 flatlines after the intersection point.

What factors could lead to this flatlining behaviour?

- Insufficient number of epochs for depths to diverge
- Vanishing gradients
- Insufficient quantity of data
- Similarly expressive model (doubtful)

![](grad_graph.png)
Aside from a turbulent beginning, there doens't seem to be any exploding (and much less) vanishing gradients.

![](increased_epochs_graph.png)
There doesn't appear to be any noticeable difference in performance across varying epochs. The flatline effect is very real.

- [x] Make sure image normalization is good
- [x] Make sure parameter initialization is good
- [x] Observe difference that clip grad norm makes

![](clipped_grad_graph.png)

There doesn't appear to be any change in performance. The difference in performance of the first epoch was in fact more pronounced, but the two changes seem to be generally applicable enough that it's probably worth keeping.

According to the [Revisiting ResNets](https://arxiv.org/pdf/2103.07579.pdf) paper, there should be a marignal but noticeable difference in error between depths 5 and 9. On ImageNet, it's a difference of under 5% in magnitude. 


## Solution