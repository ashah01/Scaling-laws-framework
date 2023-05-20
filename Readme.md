# Scaling Laws Framework

This repository serves as an opportunity for me to learn about how best to conduct scaling laws, using the classic MNIST dataset. The intention is to use this quick feedback loop environment to build a framework to be applied to more complex problems. 

Scaling laws fundamentally consists of changing the depth and width of a model, plotting its test loss, and then observing a functional form relating the variables. The primary goal should be optimizing the accuracy of your law's predictive performance. Consider various [[Scaling strategies]] (ex. scaling width exclusively, depth exclusively, combination of both, etc.). Explore good nuisance HPs values for certain regimes by exploring changes in performance across several HP values. It's also worth noting whether the values that work for one regime translate well into others. Eventually, you figure out what's the right way to scale it up such that it performs best.

Todo
----
- [x] Observe effect of increasing regularization
- [ ] Modify architecture to better accomodate very deep training (ex. use residual MLP $h = h_{\text{prev}} + \text{relu}(Wh_{\text{prev}} + b)$)
![architecture](attachments/2023-05-18-16-49-52.png)
  - [x] Batch Norm
  - [ ] Architecture

    - The basic block here seems to be FC -> ReLU -> BN -> Dropout
    - The first skip connection sources at the input tensor and skips to just after the second FC layer
    - The second skip connection sources at the third block's ReLU layer and skips to just after the fifth's FC layer
    - The third skip connection sources at the sixth block's ReLU layer and skips to just after the eighth's FC layer
    - The fourth skip connection sources at the ninth block's ReLU layer and skips to just after the eleventh's FC layer
    - Downsampling block at the end seems to be FC -> ReLU -> FC -> Softmax

  - [ ] Skip connections
    