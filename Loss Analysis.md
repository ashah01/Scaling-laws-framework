Even after doing a shallow optimization over the nuisance hyperparameters, all architectures yield worse loss as they scale past 2 layers. This must be resolved to ensure validity of the results and if similar behaviour is observed at greater scale.

- [ ] Solve loss vs depth abnormality
  - [x] Is it the vanishing gradient issue?

    Maybe try checking for vanishing gradients?
    Compare torch.norm() of gradients in 2 layer and 3 layer model

    fixed: lr at 3e-4, bsz at 32, width at 32

    **Two layers**

    loss: 0.16725250258156582

    \# of parameters: 26506
    gradient norm sum: 3.1793

    ```
    grad_sum = 0
    for l in net.layers:
        if isinstance(l, nn.Linear):
            grad_sum += torch.norm(l.weight.grad)
            grad_sum += torch.norm(l.bias.grad)
    ```

    **Three layers**

    loss: 0.17372689037375486

    \# of parameters: 27562
    gradient norm sum: 3.9237

    **Four layers**

    loss: 0.19007668130400296

    \# of parameters: 28618
    gradient norm sum: 4.9065


    *Vanishing gradient doesn't seem to be the problem.*



- [ ] Rerun shallow grid search to extract heuristics