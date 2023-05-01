# Scaling Laws Framework

This repository serves as an opportunity for me to learn about how best to conduct scaling laws, using the classic MNIST dataset. The intention is to use this quick feedback loop environment to build a framework to be applied to more complex problems

1. Choose model and dataset
2. Ensure training algorithm converges
3. Enforce early stopping (rerun epochs until test loss gets worse given some patience)
4. Optimize over nuisance hyperparameters (ex. learning rate, dropout, etc.) for each approximate model size
5. Run models on all possible shapes using grid search bash script
6. Plot loss vs. each scientific hyperparameter
7. Extrapolate law


# Research
The model and training loop has been constructed, and nuisance hyperparameter values have been chosen.
- [x] Implement early stopping to extract maximum performance from each architecture
- [x] Run grid search. Learning rate shouldn't differ since the models don't vary much in scale (embed dim 10-20, depth 2-5 size yields between 100,000-300,000 parameters. 3e-4 seems to be optimal). As per the Scaling Laws paper, don't tune dropout beyond what's been shown to work well (0.1 dropout for each)
- [ ] Plot embed dim and depth versus loss
- [ ] Extrapolate law
- [ ] Read the Greg Yang paper on scaling laws, might have some good insights about tuning the scaling laws framework above