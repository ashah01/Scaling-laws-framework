# Achieve stability

As of now, a training regime has been finalized. Learning rate and weight decay sweeps have been conducted, setting a good baseline for future scaling strategies.

![W&B results towards final steps of training regime](2023-07-09-13-06-43.png)
However, upon observing the depth scaling strategy, results are ambiguous. There's no clear best model, and worse yet, performance does not follow any clear trend (the expected one of course being an increase in performance as depth is scaled). 

This is a massive issue because the core feature of scaling laws research is the predictive power of the functional form. If every epoch regime yields a different functional form (even regimes within 1 epoch of another), the predictive power is essentially zero. We're looking for <ins>stable</ins> learning curves with clear monotonic performance trends. Larger models should be consistently more performant than smaller ones, and the observed forms shouldn't change much for epoch regimes within reasonable bounds.