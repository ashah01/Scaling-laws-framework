# Scaling Laws Framework

This repository serves as an opportunity for me to learn about how best to conduct scaling laws, using the classic MNIST dataset. The intention is to use this quick feedback loop environment to build a framework to be applied to more complex problems

1. Choose model and dataset
2. Ensure training algorithm converges
3. Enforce early stopping (rerun epochs until test loss gets worse given some patience)
4. Optimize over nuisance hyperparameters (ex. learning rate, dropout, etc.) ($\mu$Transfer might be helpful here)
5. Run models on all possible shapes using grid search bash script
6. Plot loss vs. each scientific hyperparameter
7. Extrapolate law


# Research
The model and training loop has been constructed, and nuisance hyperparameter values have been chosen.
- [x] Implement early stopping to extract maximum performance from each architecture
- [x] Run grid search. Learning rate shouldn't differ since the models don't vary much in scale (embed dim 10-20, depth 2-5 size yields between 100,000-300,000 parameters. 3e-4 seems to be optimal). As per the Scaling Laws paper, don't tune dropout beyond what's been shown to work well (0.1 dropout for each)
- [x] Plot embed dim and depth versus loss
- [x] Read the Greg Yang paper on scaling laws, might have some good insights about tuning the scaling laws framework above
- [ ] Gather more data points. Optimize nuisance hyperparameters (visualizing train vs test might be helpful for determining dropout)
  - [ ] Resolve irregularities / ensure correctness. Identify and resolve abnormalities in performance curves.
- [ ] Extrapolate law


### Why isn't depth 5 width 30 better than its shallower, slimmer counterparts?

![image](depth5width30lr8e5.png)

The performance curve seems to have immense deviation. The average loss reported is dramatically larger than those part of the lowest bound. It might be helpful to examine other learning rates and other batch sizes to determine what the variance is looking like, and see if another hyperparameter configuration is more well-suited.

Variance of test losses on batch size 64 is 0.0020
Variance of test losses on batch size 32 is 0.0043
Variance of test losses on batch size 32, lr 3e-4 is 0.0045 (has best performance)

It would seem the heuristic of smooth optimization surfaces doesn't really hold here. Smaller batch sizes with a larger learning rate are optimal. 