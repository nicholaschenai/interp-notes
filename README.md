# Brief notes on Interpretability
Literature review + quick notes for my own reference so pardon the untidyness. Will occasionally copy paste directly from the papers!

This repo has a focus on mechanistic and developmental interpretability, and occasionally there are some works that feature both of these.

---
# Developmental Interpretability
From the **evolution** of the network across training, can we identify interesting phenomena (eg phase transitions)? Singularities in loss landscape: Large gradients, this can dominate learning!

### Quantifying degeneracy in singular models via the learning coefficient
[[Repo](https://github.com/edmundlth/scalable_learning_coefficient_with_sgld/tree/v1.0)]
[[Paper](https://arxiv.org/abs/2308.12108)]
- Core idea: $\lambda$ is some invariant of the NN so can be useful indicator for phase transition or other phenomena
- local learning coeff $\lambda$ and local singular fluctuation $\nu$ for some param $\omega^*$
    - correspond roughly to complexity n functional diversity
    - their work: provide estimator for $\lambda$ that preserves ordering $\hat{\lambda}(\omega_A) < \hat{\lambda}(\omega_B)$ when $\lambda(\omega_A) < \lambda(\omega_B)$
- show that their measure can capture the degeneracy caused by entropy-SGD over regular SGD
- "we conjecture that the nature of the degeneracy of DNNs might be the secret sauce behind their state-of-the-art performance on a wide range of tasks."
- "The main application of the local singular fluctuation $\nu$ is in connection with phase transitions and will be treated in future work"
- $\lambda$ is the smallest pole of the $\zeta$ fn, with multiplicity $m$
- Definition 2: under some conditions, $\lambda$ is a birational invariant of $W_0$ (a set of optimal params) known in algebraic geom as the Real Log Cannonical Threshold (RLCT)
- for regular models (unique params, which is totally not what NNs are), $\lambda=d/2, m=1$
- $\nu$ is the expected value of a functional variance in the large data limit
- hessian estimates are wrong, need rethinking
- BIC is the first 2 terms of the laplace approx of free energy
    - however, theres sth not valid about this expansion, but this is corrected in watanabe 2009 and is one of the major theorems in SLT
- thm 4 of watanabe 2013 proposes "widely applicable BIC" (WBIC), a good estimator of free energy and does so at single temperature (other methods need to be computed across multiple temperatures)

------
# Mechanistic Interpretability
From weights, can we identify explainable algorithms?

### Progress measures for grokking via mechanistic interpretability
[[Repo](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper)]
[[Paper](https://arxiv.org/abs/2301.05217)]
- Grokking: a phenomena where test loss increases / plateaus initially while the train loss decreases, then suddenly drops many steps after the train loss has already stabilized
- mechanistic interpretability to obtain progress measures that can track underlying phenomena behind grokking on a modular addition task. They showed that a network goes through these phases:
    - Memorization: Train loss decreases sharply at the beginning.
    - Circuit formation: Though the train and test losses plateaus, the excluded loss (loss when a modification to the logits is meant to penalize generalized solutions significantly and less so on memorized solutions) increases, showing that the network is starting to generalize
    - Cleanup: Test loss and sum of squared weights sharply drops, showing that the model generalizes while the memorized solution is pruned away in the weights via weight decay
- Appendix E2: how is it possible that neural networks, which are fundamentally continuous, are able to learn discrete algorithms?
    - lottery ticket hypothesis  (for any network, we can find a subnetwork as good as the full network) suggests early on,  the neural network is a superposition of circuits, grad small cos each part of circuit is rough
    -  gradient descent slowly boosts the useful circuits until at least one of them develops sufficiently, causing other components to be more useful, so all gradients will increase together nonlinearly
    - this is alr shown for induction heads for grad of loss (Olsson et. al. 2022 "In-context Learning ... ")

### Grokking Tickets: Lottery Tickets Accelerate Grokking
[[Repo](https://github.com/gouki510/Grokking-Tickets)]
[[Paper](https://arxiv.org/abs/2310.19470)]
- Lottery tickets (NNs initialized from the topology of the trained network) derived from later stages of grokking generalize faster

### Toy Models of Superposition
[[Post](https://transformer-circuits.pub/2022/toy_model/index.html)]

### In-context Learning and Induction Heads
[[Post](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)]
- giving the network one component (via the smeared key architecture) erases the phase transition

### Understanding Addition in Transformers
[[Paper](https://arxiv.org/abs/2310.13121)]
- Observation: Phase transition for each digit in integer addition
- This paper: Model trains each digit semi-independently, hence multiple phase transitions

----
# Generic

### The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
[[Paper](https://arxiv.org/abs/1803.03635)]
- for any network, we can find a subnetwork as good as the full network
- Lottery tickets weights change vastly over time, suggesting it works because of optimization/ loss landscape, not because the weights are already close to optimal


------
# Resources

### DevInterp library
[[Repo](https://github.com/timaeus-research/devinterp)]
- For now only has estimation for lambda


-------
# TODO

### Investigating the learning coefficient of modular addition: hackathon project
[[Post](https://www.lesswrong.com/posts/4v3hMuKfsGatLXPgt/investigating-the-learning-coefficient-of-modular-addition)]


### Multi-Component Learning and S-Curves
[[Post](https://www.alignmentforum.org/posts/RKDQCB6smLWgs2Mhr/multi-component-learning-and-s-curves)]

### A Mathematical Framework for Transformer Circuits
[[Post](https://transformer-circuits.pub/2021/framework/index.html)]


