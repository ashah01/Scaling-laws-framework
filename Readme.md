# Scaling Laws Framework

This repository serves as an opportunity for me to learn about how best to conduct scaling laws, using the classic MNIST dataset. The intention is to use this quick feedback loop environment to build a framework to be applied to more complex problems. 

Scaling laws fundamentally consists of chaning the depth and width of a model, plotting its best loss and then observing a funcitonal form relating the variables. The challenging part - the polish that separates professionals from amateurs - is ensuring the loss obtained truly is the *best* loss. This requires the tuning of nuisance hyperparameters - components not directly being studied but that behave differently as the components in study are.

My current implementation for this is the dumbest of the possible methods. I use grid search, which attempts every combination within a range of values. This is suboptimal because in order to find HP optima, all possible combinations need to be exhaustively tested. The key metric then for nuisance HP tuning is speed; how long does it take for this particular tuning strategy to find optima?

More is discussed about methods to achieve this in [[Loss Analysis]]

1. Choose model and dataset
2. Ensure training algorithm converges as anticipated
3. Enforce early stopping (rerun epochs until test loss gets worse given some patience)
4. Optimize over nuisance hyperparameters (ex. learning rate, dropout, etc.) ($\mu$Transfer might be helpful here)
5. Run models on all possible shapes using grid search bash script. Ensure dimensions are spaced out.
6. Plot loss vs. each scientific hyperparameter
7. Extrapolate law


