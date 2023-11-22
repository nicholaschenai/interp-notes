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

### You’re Measuring Model Complexity Wrong
[[Post](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong)]

### Towards Developmental Interpretability
[[Post](https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability)]

------
# Mechanistic Interpretability
From weights, can we identify explainable algorithms?

### Progress measures for grokking via mechanistic interpretability
[[Repo](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper)]
[[Paper](https://arxiv.org/abs/2301.05217)]
- Grokking: a phenomena where test loss increases / plateaus initially while the train loss decreases, then suddenly drops many steps after the train loss has already stabilized
    - while this paper's main experiment is on modulo addition for a specific architecture and prime number, they confirmed the existence of grokking for other architectures and primes
- Motivation: such phase transitions can be for better (eg CoT, ICL) or worse (eg reward hacking). Important to study them, especially if these transitions are sharp.
- mechanistic interpretability to reverse engineer how a transformer implements modulo addition, obtain progress measures that can track underlying phenomena behind grokking on this task. 
    - showed that embedding maps inputs to sines and cosines at a sparse set of key frequencies
    - progress measures
        - restricted loss: ablate every non-key frequency 
        - excluded loss: ablate all key frequencies
    - Note: previous work defines progress measures heuristically while this work discovers progress measures empirically via mech interp
- Section 3: authors claim the exact algo used by the network. Section 4: studies that suggest section 3 is true, including noting periodicity in weights and activations, how the model weights compose the inputs, and ablations. These techniques are quite specific to the modulo addition task and fourier multiplication algo, but still worth a read to get a sense of how to think about reverse engineeering a model.
- Other than the 2 losses, they also track 
    - Gini coeff of the norms of Fourier components of the ebd matrix and $W_L$, the product of the MLP output matrix and the unembedding matrix (they claim that the residual path of the MLP is not used that much)
    - L2 norm of weights
- Showed that the network goes through these phases:
    - Memorization: Train and excluded loss decreases sharply at the beginning. Test and restricted loss remain high. Gini coeff flat. Suggests that model is memorizing data, and not using the key frequencies in the fourier multiplication algo
    - Circuit formation: Though the train and test losses stay flat, the excluded loss (which is sensitive to generalized solutions and less so on memorized solutions) increases, restricted loss starts to fall, showing that the network is starting to generalize. Fall in sum sq weights (L2 norm) suggest circuit formation likely happens due to weight decay, and well before grokking!
    - Cleanup: Test loss and L2 norm of weights sharply drops, restricted loss continues to drop, showing that the model generalizes while the memorized solution is pruned away in the weights via weight decay. the sharp increase in Gini for the two matrices show that the network is becoming sparser in the Fourier basis.
- Future work
    - studies on larger models and realistic tasks
    - develop techniques to predict _when_ phase transition occurs
- Appendix misc
    - weight decay is necessary for grokking
    - amount of data affects grokking
    - other algorithmic tasks
        - 5 digit addition: Phase transition for each digit
        - overall emphasize role of limited data for grokking
- Appendix impt (E and F)
    - E1: intuitive explanation of grokking (speculation)
        - 2 competing forces: memorization and generalization
            - with little data, model will overfit and memorize
            - with more data, model must generalize to decrease loss
            - NNs have inductive bias favoring simpler solns (compounded by regularization?)
            - Memorization complexity scales with size of train set, but generalization complexity is constant -- there will eventually be a crossover
        - phase transitions: why is grokking abrupt? suggests that generalizing soln has some 'barrier' / threshold to overcome rather than having a smooth gradient to follow
        - As model memorizes, network becomes more complex until weight decay prevents further memorization
        - During generalization, model is incentivized to both memorize and simplify, and it surprisingly is capable of doing this while maintaining a constant train loss!
        - "memorization is not necessarily a “simpler” solution than generalization. The key is that generalization will have smaller weights _holding train loss fixed_"
    - E2: HYPOTHESIS: PHASE TRANSITIONS ARE INHERENT TO COMPOSITION
        - Big qn: how is it possible that neural networks, which are fundamentally continuous, are able to learn discrete algorithms?
            - Furthermore, how is it possible since in a multi-component circuit, one component is useless without the other(s)? (how do we even get gradients?)
        - possible explanations:
            - lottery ticket hypothesis (for any network, we can find a subnetwork as good as the full network) has the best explanation
                - suggests early on,  the neural network is a superposition of circuits, grad small cos each part of circuit is rough
                -  gradient descent slowly boosts the useful circuits until at least one of them develops sufficiently, causing other components to be more useful, so all gradients will increase together nonlinearly
                - this is alr shown for induction heads for grad of loss (Olsson et. al. 2022 "In-context Learning ... ")
            - random walk
                - insufficient explanation as an induction circuit is relatively complicated
            - evolution: one component somehow develops independently as it is generically useful, then other components follow
                - cant be the whole story. In their repeated subsequence experiments, the 2 components require each other to be useful
    - F: How can mech interp and progress measures help us understand/ predict emergent phenomena in the future
        - If mech interp can be scaled to large models, then we can use the same style of analysis in this paper to get progress measures
        - If mech interp only explains larger models partially, we can still use our understanding from smaller models to guide development measures that track parts of the larger model
        - if mech interp fails to recover understandable mechanisms on large models, we might still be able to derive progress measures on an opaque component, say via automated mech interp methods

### Grokking Tickets: Lottery Tickets Accelerate Grokking
[[Repo](https://github.com/gouki510/Grokking-Tickets)]
[[Paper](https://arxiv.org/abs/2310.19470)]
- This paper (and thus these notes) are written in the context of "Progress measures for grokking via mechanistic interpretability", read that first! Also get a general idea of the lottery ticket hypothesis (see notes below)
- Investigates the role of lottery ticket hypothesis in grokking by deriving lottery tickets from various methods and phases of grokking, and comparing their accuracy/ loss as we train them
- Architectures: MLP and Transformer (minor note: this transformer does not have biases in the MLP layer, unlike the original progress measures paper)
- tasks: modulo addition and image classification
- Denote lottery tickets corresponding to the generalized solution (post-cleanup) as _grokking tickets_
- 4 main comparisons
    - base model (no modifications)
    - non-grokking tickets: lottery tickets derived from stages before generalization
    - controlled dense model with weights of same L1 or L2 norm as grokking tickets
    - Pruning-at-initialization tickets to control for sparsity
- main pruning strategy: magnitude of weights. (pruning-at-initialization uses other separate strategies)
- Main results
    - The later the epoch which the lottery ticket is derived from, the faster the lottery ticket generalizes (and thus the test accuracy at a fixed epoch of training follows an increasing trend wrt the epoch that the ticket is derived from)
    - grokking tickets generalize faster than controlled models of same L1/L2 weight norm
    - pruning at initialization methods (SNIP, Grasp, Synflow, random) either do not grok or generalize slower than base model
        - "The results show that poor selection of the subnetworks hurts the generalization, suggesting that grokking tickets hold good properties beyond just sparsity"
    - no generalization speedup when using tickets from memorization stage, transition between memorization and generalization (cleanup stage)
    - grokking occurs for various pruning ratios, except when it is too high (eg 0.9 where the accuracy gets stuck around 0.4, indicating lack of capacity in NN)
    - when using a grokking ticket with the largest pruning ratio that allows for grokking (pruning ratio of 0.81, obtained via param sweep between 0.8 and 0.9), grokking occurs even without weight decay!
        - "suggesting that weight decay is crucial for uncovering the lottery tickets but becomes redundant after their discovery"
        - note that this critical pruning ratio of 0.81 is important because if weight decay is important, then it might prune away some weights to increase the pruning ratio and thus disable grokking
        - note: this pheonmenon only for MLP. In transformers, no weight decay disables grokking
- "results imply that the transition between the memorized and generalized solution corresponds to exploring good subnetworks, i.e., lottery tickets."
- Important appendix stuff
    - E: Confirming that grokking tickets give good representations
        - plot effective in and out weight matrices for base and grokking ticket models in the modulo addition task: find that grokking tickets display the periodic patterns at an earlier epoch!
        - plot discrete Fourier transform of the above: find that the sharp frequency peaks are obtained at an earlier epoch for grokking tickets
        - define frequency entropy, plot them for the 2 models: grokking tickets' entropy decreases much faster than base model, favoring simpler frequency characteristics for both in and out weight matrices


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
- formally: "dense, randomly-initialized, feed-forward networks contain subnetworks (winning tickets) that—when trained in isolation—reach test accuracy comparable to the original network in a similar number of iterations."
- Algo
    - randomly init a NN $f(x;\theta_0)$
    - train network for $j$ iterations to arrive at params $\theta_j$
    - prune a fraction of the params in $\theta_j$ to obtain a mask $m$
    - reset params, creating the winning ticket $f(x;\theta_0 \odot m)$
- they explore various pruning strategies
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


