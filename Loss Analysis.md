Even after doing a shallow optimization over the nuisance hyperparameters, all architectures yield worse loss as they scale past 2 layers. This must be resolved to ensure validity of the results and if similar behaviour is observed at greater scale.

- [x] Solve loss vs depth abnormality
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

  - [x] Is it just simple overfitting?

    I'm doubtful because increasing dropout only negatively impacts the model, but maybe some other kind of regularization needs to be put in place? With the depth set to 4 layers, the training loss for the final epoch averages to 0.13917233482835192 while the test loss averages to 0.1779356023857218. With the depth set to 2 layers, the training loss for the final eopch averages to 0.1712026208177209 while the test loss averages to 0.18512281320078638. Maybe try larger depths with slightly more dropout?

    Depth 4 with 0.055 regularization yields train loss 0.16042481156115732 and test 0.19096997898063078
    Depth 4 with 0.06 regularization yields train loss 0.1654005287251125 and test 0.21014901985393497

    How does training loss compare across depth? If that's not even greater, no amount of regularization can help us
    Depth 2 with 0.06 dropout yields train loss 0.147692727419734 and test 0.16831565281925218???
    Depth 2 with 0.05 dropout yields train loss 0.14235558503915866 and test 0.1696983874411346
    Depth 4 with 0.05 dropout yields train loss 0.15041534943605464 and test 0.17945787891747056

    Ok something is deeply fundamentally wrong. Figure it out.

    Ahh stupid ass dropout blocked too much signal, scaled regularization faster than model size.


- [ ] Build small scale configurator applying learned heuristics