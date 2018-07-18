GANs
====

+ [GAN](./travialgan.py): Generative Adversarial Nets, [NIPS 2014](https://papers.nips.cc/paper/5423-generative-adversarial-nets), [arxiv](https://arxiv.org/abs/1406.2661)

    + Loss_d = E[\log{D(x)}] + E[\log{1 - D(G(z))}]
    + loss_g = E[\log{1 - D(G(x))}]

+ [DCGAN](./dcgan.py): Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016, [arxiv](http://arxiv.org/abs/1511.06434)

    + CNNs: replace deterministic spatial pooling functions (such as maxpooling) with
      strided convolutions.
    + Eliminating fully connected layers on top of convolutional features.
      + Directly connect the highest convolutional features to the input and output
        respectively of the generator and discriminator.
    + Batch normalization, but not applying to the generator output layer and discriminator
      input layer. Otherwise resulting to sample oscillation and model instability.
    + ReLU activation in generator: using a bounded activation allowed the model to learn
      more quickly.
    + LeakyReLU activation in discriminator.

+ [LSGAN](./lsgan.py): Least Squares Generative Adversarial Networks, [ICCV 2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf), [arxiv](https://arxiv.org/abs/1611.04076)

    + Fix the instability of GANs learning
    + Least square criterion: penalizes samples that lie in a long way on the correct
      side of the decision boundary.

+ [WGAN](./wgan.py): Wasserstein Generative Adversarial Networks, [ICML 2017](http://proceedings.mlr.press/v70/arjovsky17a.html), [arxiv](https://arxiv.org/abs/1701.07875)

    + Earth-Mover (EM) distance better than other distances (KL divergence, JS divergence, etc.)
      - continuous
      - differentiable
    + W(P_r, P_\theta) = \sup{E_{x ~ P_r}[f(x)]} - E_{x ~ P_\theta}[f(x)]}
      - Equivalent to \max{E_{x ~ P_r}[f(x)]} - E_{x ~ P_\theta}[f(x)]}
    + Fixes the gradient vanishing problem (Figure 3 of the original paper)
    + Fixed the mode collapse problem, because the optimization processing is sound and optimal
    + No longer need to balance generator and discriminator's training, since WGAN trains the critic till optimality
    + Use RMSProp optimizer: the loss for the critic is nonstationary, momentum based methods perform worse
    + Weight clamping: clamp weights (after gradient update) to [-0.01, 0.01] to have parameters lie in a compact space (to enforce smoothness)
    + No sigmoid layer at the end of discriminator

+ [CGAN](./cgan.py): Conditional Generative Adversarial Nets, [arxiv](https://arxiv.org/pdf/1411.1784)

+ Tricks

    + Use non-saturating GAN loss and spectral normalization as default choice [arxiv:1807.04720]
    + Use feature matching: [arxiv:1606.03498]

        Let f(x) denote activations on an intermediate layer of the discriminator, the object
        function for generator is defined as: ||E_{x~p_{data}}{f(x)} - E_{z~p_z(z)}{f(G(z))}||_2^2.
        And the discriminator are trained in the usual way.

    + One-sided label smoothing: replaces the 0 and 1 targets for a classifier with smoothed values,
      like 0.9 and 0.1 [arxiv:1606.03498]
    + Mini-batch discrimination: TODO [arxiv:1606.03498]
