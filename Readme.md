# Scaling Laws Framework

This repository serves as an opportunity for me to learn about how best to conduct scaling laws, using the classic MNIST dataset. The intention is to use this quick feedback loop environment to build a framework to be applied to more complex problems. 

Scaling laws fundamentally consists of chaning the depth and width of a model, plotting its test loss, and then observing a funcitonal form relating the variables. The primary goal should be optimizing the accuracy of your law's predictive performance. Consider various scaling strategies (ex. scaling width exclusively, depth exclusively, combination of both, etc.). Explore good nuisance HPs values for certain regimes by exploring changes in performance across several HP values. It's also worth noting whether the values that work for one regime translate well into others. Eventually, you figure out what's the right way to scale it up such that it performs best.


1. Choose model and dataset
2. Ensure training algorithm converges as anticipated
3. Enforce early stopping (rerun epochs until test loss gets worse given some patience)
4. Optimize over nuisance hyperparameters (ex. learning rate, dropout, etc.) ($\mu$Transfer might be helpful here)
5. Run models on all possible shapes using grid search bash script. Ensure dimensions are spaced out.
6. Plot loss vs. each scientific hyperparameter
7. Extrapolate law