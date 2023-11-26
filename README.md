# Brief notes on Interpretability
Literature review + quick notes for my own reference so pardon the untidyness. Will occasionally copy paste directly from the papers!

This repo has a focus on mechanistic and developmental interpretability, and occasionally there are some works that feature both of these.

---
# Developmental Interpretability
Identify structure from phase transitions as the network **evolves** across training. 

## Intro
Based off 
[[You’re Measuring Model Complexity Wrong](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong)]
[[Towards Developmental Interpretability](https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability)]
[[Neural networks generalize because of this one weird trick](https://www.lesswrong.com/posts/fovfuFdpuEwQzJu2w/neural-networks-generalize-because-of-this-one-weird-trick)]

- Why phase transitions?
    - they exist (the most of this repo's notes are focused on phase transitions)
    - easy to find, eg emergence of circuits. 
        - Currently known observables to detect phase transitions: RLCT, singular fluctuation
        - "we don't yet know how to invent finer observables of this kind, nor do we understand the mathematical nature of the emergent order."
    - good candidates for universality
- Why Singular Learning Theory?
    - Singular models: parameters not unique (decent assumption in todays era where models are overparameterized)
        - consequence: Loss landscape looks more like a valley than a basin (lowest loss is not a single point!)
        - minimum loss sets can also look like a bunch of lines intersecting -- at this intersection, the tangent is ill-defined (singularity, hence the name for SLT!)
        - "Complex singularities make for simpler functions that generalize further."
    - In singular models, learning is dominated by phase transitions 
        - "as information from more data samples is incorporated into the network weights, the Bayesian posterior can shift suddenly between qualitatively different kinds of configurations"
        - "phase transitions can be thought of as a form of internal model selection where the Bayesian posterior selects regions of parameter space with the optimal tradeoff between accuracy and complexity"
        - Compare this sudden shifting of params vs smooth changes in params from regular gradient descent -- this is why the former is likely to dominate
    - "Phase transitions over the course of training have an unclear scientific status" but we hope SLT can explain some of them -- **Possibly related via the underlying geometry of the loss landscape**
    - If we can **relate structures in NNs (eg circuits) to changes in the geometry of singularities**: new way of thinking about these systems!
    - Additional notes
        - Lower level aim of learning: find the optimal weights for the given dataset
        - Higher lvl aim of learning:  find the optimal model class/architecture for the given dataset.
            - Bayesian learning: integrate out weights to make statements over model classes, but usually intractable
            - So use laplace approx of free energy, first 2 terms is the BIC, which balances accuracy and simplicity, i.e. regularization
            - however, assumptions for laplace approx do not hold!
                - parameter-function map is not one-to-one: can have different sets of weights that implement the same fn
                - central limit theorem does not hold
                - standard remedy: fit paraboloid to hessian of the loss landscape, add small $\epsilon$ to Hessians to make it invertible. This doesnt work in our case because the zeroes are important -- reduce the effective dimensionality of the model. 
        - (after some physics): "If we manage to find a way to express $K(w)$ as a polynomial, this lets us to pull in the powerful machinery of algebraic geometry, which studies the zeros of polynomials. We've turned our problem of probability theory and statistics into a problem of algebra and geometry."
        - Difficult to study these algebraic varieties ($W_0$, set of optimal params) close to their singularities, so must resolve them. Map them to a new manifold so that in the new coords, the singularities cross "normally"
        - need to be careful that measured quantities dont change with mapping, so use birational invariants -- in particular, the RLCT ("effective dimensionality" near the singularity)
            - More important than estimating the volume is estimating the volume dynamics and volume scaling
        - Watanabe 2009 fixes the central limit theorem to work in singular models, deriving the asymptotic free energy (limit of infinite data)
        - $\lambda$ is the smallest pole of the $\zeta$ fn, with multiplicity $m$
            - for regular models (unique params), $\lambda=d/2, m=1$. asymptotic free energy simplifies to BIC!
        - "The important observation here is that the global behavior of your model is dominated by the local behavior of its "worst" singularities. "
        - Physics: study free energy because it is a generating fn, derivatives give us useful physical quantities. Bayesian inference: Free energy generates quantities we care about, like the expected generalization loss. In both cases, we are intersted in how continuously changing some params lead to phase transitions in the quantities!
        - NNs generalize well due to their ability to exploit symmetry, but the generic ones (like $GL_n$ associated to the residual stream in transformers) are not that interesting because they are always present for any choice of $w$. We are interested in the non generic symmetries that depend on $w$. examples:
            - " degenerate node symmetry, which is the well-known case in which a weight is equal to zero and performs no work"
            - "weight annihilation symmetry in which multiple weights are non-zero but combine to have an effective weight of zero"
        - see [Intuition for phase transitions](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong#Instead__Basin_Dimension) for a nice graphic
        - SGLD: "allows random movement along low loss dimensions, while quickly pushing any step to a higher loss area back down."
- Alignment / Safety
    - **Abrupt changes are the most dangerous when it comes to alignment/ safety.**
    - Potential Use Cases 
        - Detecting deception in models
            - Need tools to distinguish deceptive vs non-deceptive models that behave similarly on evaluation, but differently on out-of-distribution data. Use invariants (of model complexity)!
        - Understanding reasoning in models to prevent dangerous scenarios
        - Detecting situation awareness in models
    - Goals in this context
        - " detecting when structural changes happen during training"
        - "localize these changes to a subset of the weights"
        - "give the changes their proper context within the broader set of computational structures in the current state of the network. "
- Challenges
    - Maybe most of the important structures form _gradually_ instead of abruptly
    - Maybe phase transitions occur so often that it is hard to discern meaningful ones from the rest
    - Maybe the transitions are irreducibly complex; i.e. there is no computational advantage in using tools to understand the model
        - Counterargument: locality in deep learning (eg lottery ticket hypothesis)
    - Maybe the structures formed in phase transitions are too disjoint to give us a good understanding of the model
    - NNs might not be doing Bayesian inference
- Roadmap
    - Apply tools to increasingly complex systems (eg induction heads)
    - For models known to contain circuits
        - use the tools to detect phase transitions
        - attempt to classify weights at each transition into state variables, control variables, and irrelevant variables
            - my note: renormalization group techniques are on their todo list, and its quite likely that this will be useful for the relevant/ irrelevant variables task?
        - mech interp at checkpoints
        - compare all these to structures found in the final model
    - Improving estimates of $\hat{\lambda}$, address weaknesses of SGLD
    - Other observables like singular fluctuation

## Papers

### Quantifying degeneracy in singular models via the learning coefficient
[[Repo](https://github.com/edmundlth/scalable_learning_coefficient_with_sgld/tree/v1.0)]
[[Paper](https://arxiv.org/abs/2308.12108)]
- Core idea: $\lambda$ is some invariant of the NN so can be useful indicator for phase transition or other phenomena
- local learning coeff $\lambda$ and local singular fluctuation $\nu$ for some param $\omega^*$
    - correspond roughly to complexity n functional diversity
    - their work: provide estimator for $\lambda$ that preserves ordering $\hat{\lambda}(\omega_A) < \hat{\lambda}(\omega_B)$ when $\lambda(\omega_A) < \lambda(\omega_B)$
    - local (restrict to subset of weights) because it is more tractable than global
- show that their measure can capture the degeneracy caused by entropy-SGD over regular SGD
- "we conjecture that the nature of the degeneracy of DNNs might be the secret sauce behind their state-of-the-art performance on a wide range of tasks."
- "The main application of the local singular fluctuation $\nu$ is in connection with phase transitions and will be treated in future work"
- Definition 2: under some conditions, $\lambda$ is a birational invariant of $W_0$ known in algebraic geom as the Real Log Cannonical Threshold (RLCT)
- $\nu$ is the expected value of a functional variance in the large data limit
- thm 4 of watanabe 2013 proposes "widely applicable BIC" (WBIC), a good estimator of free energy and does so at single temperature (other methods need to be computed across multiple temperatures)
- $\hat{\lambda}$ estimation still needs hyperparams like number of samples, localizing strength, SGLD learning rate which the true value $\lambda$ is independent of
- see [this](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong#Limitations) for limitations

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
    - when using a grokking ticket with the largest pruning ratio that allows for grokking (pruning ratio of 0.81, obtained via param sweep between 0.8 and 0.9), generalization occurs even without weight decay! (they used the term "grokking" but not sure if its the right word as the generalization is immediate)
        - "suggesting that weight decay is crucial for uncovering the lottery tickets but becomes redundant after their discovery"
        - note that this critical pruning ratio of 0.81 is important because if weight decay is important, then it might prune away some weights to increase the pruning ratio and thus disable generalization
        - note: this pheonmenon only for MLP, they did not observe the same in transformers
        - **BIG NOTE** I observe that the test accuracy of the grokking ticket without weight decay plateaus **below** the base model, suggesting that the lack of weight decay can still hurt performance
- "results imply that the transition between the memorized and generalized solution corresponds to exploring good subnetworks, i.e., lottery tickets."
- Important appendix stuff
    - E: Confirming that grokking tickets give good representations
        - plot effective in and out weight matrices for base and grokking ticket models in the modulo addition task: find that grokking tickets display the periodic patterns at an earlier epoch!
        - plot discrete Fourier transform of the above: find that the sharp frequency peaks are obtained at an earlier epoch for grokking tickets
        - define frequency entropy, plot them for the 2 models: grokking tickets' entropy decreases much faster than base model, favoring simpler frequency characteristics for both in and out weight matrices

### In-context Learning and Induction Heads
[[Post](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)]
- Induction Heads: circuit that copies and completes sequences that have occurred before. eg if the sequence is `... [A][B]......[A]`, then the induction head predicts `[B]`
- "Mechanically, induction heads in our models are implemented by a circuit of two attention heads:"
    - 'the first head is a “previous token head” which copies information from the previous token into the next token'
    - 'the second head (the actual “induction head”) uses that information to find tokens preceded by the present token.' 
    - "For 2-layer attention-only models, we were able to show precisely that induction heads implement this pattern copying behavior and appear to be the primary source of in-context learning."

- Hypothesis: "induction heads might constitute the mechanism for the actual majority of all in-context learning in large transformer models" 
    - how? show that the phase change (visible bump in training loss) early in training for language models of various size relates to the acquisition of ICL, and show that it is causal (eg intentionally causing the bump to appear at different epochs of training causes ICL ability to be acquired around that new epoch). (note: this is about general ICL, not just literal copying of tokens)
    - evidence is strong and causal for small models, but medium and correlational for large models
- Per-Token Loss Analysis allows us to analyze and compare training trajectories
    - run model / snapshot over a set of examples, collecting one token's loss per example
    - for each sample, extract loss of a consistent token and combine them into a vector
    - analyze vector by PCA and 2D projection
- Heuristic measure of ICL ("ICL score"): loss of 500th token - avg loss of 50th token in the context, averaged over dataset examples. (authors also varied the numbers to show that conclusions do not change)

1. "Transformer language models undergo a “phase change” early in training, during which induction heads form and simultaneously in-context learning improves dramatically."
    - medium, correlational evidence across model sizes
    - models with > 1 layer exhibit phase transition in ICL score 
        - during phase transition, derivative of loss wrt token index turns negative for such models even for large token indices, i.e. models continue to reduce loss with longer context.
        - prefix matching score abruptly increases during this transition, further supporting the hypothesis
        - loss curves display a bump during this period but it is quite small
        - the 2D plot of the first and second principal component (Per-Token Loss Analysis) shows a change in trajectory during this phase transition
    - caution
        - for larger models, time resolution is lower (15 data points for the phase transition) so evidence is weaker
        - correlation != causation as usual. might be a shared latent variable
2. " When we change the transformer architecture in a way that shifts whether induction heads can form (and when), the dramatic improvement in in-context learning shifts in a precisely matching way."
    - medium, interventional evidence for small models, weak, interventional evidence for large models
    - smeared key architecture: give the network one of two components of an induction head
        - one layer model now displays phase change in ICL score! recall that one layer model might struggle to express 2 components, but giving it one component and seeing the phase transition appear suggests that induction heads play a role in ICL
        - for models > 1 layer, the phase change occurs much earlier after giving it the smeared key arch
3. Ablating induction heads decreases ICL
    - strong-med causal evidence for small models
    - ablate over different types of heads; ablating induction heads generally cause worse performance, but not so for other types of heads!
4. Empirical observation that induction heads " also appear to implement more sophisticated types of in-context learning, including highly abstract behaviors"
    - plausible for large models
    - Behavior 1: Literal sequence copying
    - Behavior 2: Translation
    - Behavior 3: Pattern matching
5. "For small models, we can explain mechanistically how induction heads work, and can show they contribute to in-context learning. Furthermore, the actual mechanism of operation suggests natural ways in which it could be re-purposed to perform more general in-context learning."
    - contributes some: strong, mechanistic for small models, medium, mechanistic for large models
    - can reverse engineer induction heads in transformers but not those with MLP layers
    - see [their previous work](https://transformer-circuits.pub/2021/framework/index.html) for reverse engineering induction heads
6. "Extrapolation from small models suggests induction heads are responsible for the majority of in-context learning in large models."
    - model analysis table to show how some behaviors persist when varying from small to large models
    - there are cases where large models behave differently from small models, so extrapolate with caution
        - eg other composition mechanisms may form during the phase change as larger models have more heads to do so
        - 'If all “composition heads” form simultaneously during the phase change, then it’s possible that above some size, non-induction composition heads could together account for more of the phase change and in-context learning improvement than induction heads do.'
- Unexplained curiosities
    - Seemingly constant ICL score after phase change, across all models
    - The ordering of derivative of loss wrt log(train tokens) for models of various sizes invert at phase change! (originally small models improve faster, but after phase change, large models improve faster)
- Additional Curiosities
    - 6-layer attention-only model has a head that is not an induction head but ablating it has similar effect to reversing the phase change
    - "4-layer MLP model ablations are nowhere near as “peaky” as those of any other model"
    - "6-layer MLP model shows a “loss spike”"
    -  "6-layer MLP model has one lone induction head whose ablation has the opposite effect on the in-context learning score"
    - "Full-scale models above 16 layers start to show a small number of heads that score well on “prefix search”, but get a negative score on copying, which means they are not induction heads"
- Discussion
    - studying ICL is impt for safety as "model behavior can in some sense “change” during inference, without further training", and this behavior can be unwanted
    - "We did not observe any evidence of mesa-optimizers."
- Other
    - "Circuits thread tried to extend this notion of universality from features to circuits, finding that not only do at least some families of well-characterized neurons reoccur across multiple networks of different architectures and that the same circuits, but the same circuits appear to implement them"

### Toy Models of Superposition
[[Post](https://transformer-circuits.pub/2022/toy_model/index.html)]
- Definitions
    - Toy models: small ReLU networks trained on synthetic data with sparse input features
    - Superposition: models represent more features than dimensions
    - Privileged basis: when features align with basis directions (correspond to neurons), as opposed to superposition where features are not aligned (think of it as, features can have the $[1,0], [0,1]$ privileged basis vs $\frac{1}{\sqrt(2)}[1,1], \frac{1}{\sqrt(2)}[1,-1]$ basis)
        - need something to break symmetry to use privileged basis (eg activation fn)
        - eg text ebds use non-privileged basis as they have no reason to break symmetry, so even the same type of ebd trained in a different run can produce different vectors as they are similar up to a rotation
        - eg CNNs tend to have privileged basis as the inputs have spatial structure
    - polysemantic neurons: appear to respond to unrelated mixtures of inputs
- Hypotheses
    - linear representation hypothesis (they believe this to be widespread)
        - NN representations can be decomposed into understandable features
        - linearity: features are represented by direction
    - feature properties that occur sometimes:
        - non-superposition, i.e. $W^TW$ is invertible
        - basis-aligned: all $W_i$ are one-hot vectors. partially basis aligned if $W_i$ sparse. requires privileged basis
    - Superposition Hypothesis: 'neural networks "want to represent more features than they have neurons", so they exploit a property of high-dimensional spaces to simulate a model with many more neurons.'. mathematical motivation:
        - Johnson–Lindenstrauss lemma allows exp(n) "almost orthogonal" vectors in high-dimensional spaces
        - compressed sensing: if vector is sparse, and is projected into a lower-dim space, the original vector can be reconstructed. not true for dense vectors
    - Linear fns in NNs are responsible for most of the computation, so linear repr are the natural format for NNs
        - one advantage is the consistency which a neuron can select a feature in the previous layer 
        - "Representing features as different directions may allow non-local generalization in models with linear transformations (such as the weights of neural nets), increasing their statistical efficiency relative to models which can only locally generalize. "
- Key results
    - Evidence for superposition in their toy models, suggesting it might also appear in realistic models. "exploring how it interacts with privileged bases. If superposition occurs in networks, it deeply influences what approaches to interpretability research make sense, so unambiguous demonstration seems important."
    - Both monosemantic and polysemantic neurons can form.
    - At least some kinds of computation can be performed in superposition.
    - Phase changes in various forms, eg whether features are stored in superposition or not, dimensionality of features
    - (might be specific to toy models) Superposition organizes features into geometric structures such as digons, triangles, pentagons, and tetrahedrons.
    - "We find preliminary evidence that superposition may be linked to adversarial examples and grokking, and might also suggest a theory for the performance of mixture of experts models"
- Demonstrating superposition
    - Experiment: Autoencoder reconstruction
    - Assumptions
        - Feature sparsity (rarely occur). eg tokens in language, curves in various locations of an image
        - more features than neurons
        - features vary in importance
    - loss: reconstruction MSE but weighted by feature importance $I_i$
    - result: as we move from the dense to sparse regime, the model moves from representing only the top $m$ (bottleneck dim) important features and positive biases to representing nearly all the $n$ (input dim) features but with interference / superposition and negative biases
    - Mathematical understanding: binomial expansion and re-express the loss around $S$, obtaining terms of various orders. As $S$ tends to 1 (sparse regime), only the first 2 loss terms dominate. 0th term isnt interesting but 1st term is: similar to loss of linear neural networks but affected by bias-- result: interference term can have lower loss as negative bias can mitigate interference term! 
    - related to Thomson problem in chemistry: arranging electrons on the surface of a sphere to minimize energy
    - increased sparsity results in lesser interference (similar concept to collisions?), so the benefit of more features (thus increased accuracy) can outweigh the interference penalties
- Superposition as a Phase Change
    - phase diagram for $n=2,m=1$ with feature importance and feature density as axes: 3 regimes where extra feature is not represented, gets dedicated dim, gets stored in superposition!
        - dense regime: feature goes from not represented to dedicated dim as feature importance increases
        - sparse regime: feature is stored in superposition, always
        - intermediate regime: not represented -> superposition -> dedicated dim as feature importance increases
        - theory and expt match!
    - $n=3, m=2$ more complicated, now have 4 cases: not represented, superposition, dedicated dim but others not represented, dedicated dim but others in superposition. see fig
- The Geometry of Superposition
    - Uniform Superposition, where all features are identical and independent
        - plot $m/ \left\lVert W  \right\rVert_F $ ("dimensions per feature" since heuristically the weight is 1 if a feature is represented and 0 if it is not)
            - as we go from the dense to sparse regime, this decreases but has 2 'sticky plateaus' at 1 and 1/2 which vaguely resembles FQHE
            - 1/2 corresponds to regime where features come in antipodal pairs (one negative of the other)
        - define the $i$th feature dimensionality as the L2 norm of the vector as a fraction of the sum sq of the projections of other features onto the unit $i$th feature. denominator represents "how many features share the dimension it is embedded in"
        - from the dimensions per feature plot, overlay with scatter plot for feature dimensionality of all features
            - see figure
            - features cluster (i.e. many interfere with each other) at specific fractions of feature dimensionality
                - with interesting geometry! similar to Thomson problem
                    - solns can be understood as tegum products
                    - "possible reason why we observe 3D Thomson problem solutions, despite the fact that we're actually studying a higher dimensional version of the problem. Just as many 3D Thomson solutions are tegum products of 2D and 1D solutions, perhaps higher dimensional solutions are often tegum products of 1D, 2D, and 3D solutions."
                    - "The orthogonality of factors in tegum products has interesting implications. For the purposes of superposition, it means that there can't be any "interference" across tegum-factors."
            - "Superposition is what happens when features have fractional dimensionality. That is to say – superposition isn't just one thing!"
    - Non-Uniform Superposition
        - far from comphrehensive theory of geometry for this, so they highlight some of the more striking phenomena
            - varying importance/ sparsity "causes smooth deformation of polytopes as the imbalance builds, up until a critical breaking point at which they snap to another polytope."
                - eg for $n=5,m=2$, phase transition from digon to pentagon. plotting loss curves for both solns show that during the phase transition, the loss curves switch rankings
            - "Correlated features prefer to be orthogonal, often forming in different tegum factors."
                - see graphics!
                - "Conversely, when features are anti-correlated, models prefer to have them interfere, especially with negative interference"
                - In some cases, correlated features merge into their principal component $(a+b)/\sqrt{2}$ 
                    - when going from sparse to dense regime, the features go from superposition to principal component
                    - "this hints at some kind of interaction between "superposition-like behavior" and "PCA-like behavior"."
                - "models prefer to arrange correlated features side by side if they cant be orthogonal"
            - "Anti-correlated features prefer to be in the same tegum factor when superposition is necessary. They prefer to have negative interference, ideally being antipodal."
    - might not generalize to larger models
- Superposition and Learning Dynamics
    - Discrete jumps
        - use model where all features converge into digons. as training progresses, feature dimensionality
            - starts in a band between 0-0.3
            - then bifurcates to head towards 0 or 0.5 and settle there (attractors?). loss was decreasing and plateaus after the feature dim settles
            - occasionally some features abruptly jump between the 0, 1/2, $1/\sqrt{2}$ level! loss drops abruptly after this
    - "LEARNING AS GEOMETRIC TRANSFORMATIONS"
        - see graphic
    - might not generalize to larger models
- Experiments show that "vulnerability to adversarial examples sharply increases as superposition forms" and the level of vulnerability closely tracks the inverse feature dimensionality
- Using ReLU can influence the model to use a privileged basis, allowing weights to be directly interpretable. Done so for a toy model to show how increasing sparsity changes neurons from monosemantic to polysemantic, but unclear how this generalizes to real models as the model doesnt benefit from the ReLU hidden layer
- Computation in Superposition
    - task: train a bottlenecked network with 1 hidden layer to learn the absolute value fn. vary sparsity
    - result: from dense to sparse regime, neurons start from monosemantic to increasing numbers of polysemantic neurons
        - degree of polysemanticity also increases. in the intermediate regime, polysemantic neurons are still strong in 1 feature but weak in other features, while in the sparse regime, the polysemantic neurons have nontrivial components in multiple features
            - this intermediate regime neurons are found in realistic language models
    - observation 1: asymmetric superposition (i.e. magnitudes of features are not the same) to have different magnitudes of interference (see graphic)
    - observation 2: inhibition! (note, biological neurons also inhibit each other)
    - "This leads us to hypothesize that the neural networks we observe in practice are in some sense noisily simulating larger, highly sparse networks."
- The Strategic Picture of Superposition
    - if there is a technique to project superposition models back to their higher dim non-superposition form, we can identify and enumerate over features, which is important for safety and interpretability. this is equivalent to or necessary for
        - Decomposing Activation Space
        - Describing Activations in Terms of Pure Features
        - circuit analysis
        - disentangling basic approaches, e.g. we use cosine similarity a lot but in superposition, unrelated features can also have positive dot products which confounds our analysis
    - challenge: real, performant models tend to employ superposition as it is useful, so getting rid of it is hard but we might try understanding phase transitions to have a model be in the non-superposition regime?
    - "At present, it's challenging to study superposition in models because we have no ground truth for what the features are."
    - "Local bases are not enough." " if our goal is to eventually make useful statements about the safety of models, we need mechanistic accounts that hold for the full distribution (and off distribution). Local bases seem unlikely to give this to us."
- Discussion
    - Polysemantic neurons exist in the wild, eg InceptionV1 has more polysemantic neurons in later layers and "Early Transformer MLP neurons are extremely polysemantic"
    - "it's very unclear what to generalize to real networks." (from toy models)
    - open qns
        - "Is there a statistical test for catching superposition?"
        - "How can we control whether superposition and polysemanticity occur?"
        - "Should we expect superposition to go away if we just scale enough?"
        - "How important are polysemantic neurons?"
        - "Does the apparent phase change we observe in features/neurons have any connection to phase changes in compressed sensing?"
        - "How does superposition relate to non-robust features? "
        - "To what extent can neural networks "do useful computation" on features in superposition?"
        - "Can models effectively use nonlinear representations?"
- Other
    - the relation (and differences with) compressed sensing is worth a read

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
- tip: "sweep over SGLD LR and localization strength, be as large as possible and visualize trace of losses, should grow n plateau. Tune to FINAL weights"

-------
# TODO

### Dynamical and Bayesian Phase Transitions in a Toy Model of Superposition 
[[Paper](https://arxiv.org/abs/2310.06301)]
- Also see explainer [[Growth and Form in a Toy Model of Superposition](https://www.lesswrong.com/posts/jvGqQGDrYzZM4MyaN/growth-and-form-in-a-toy-model-of-superposition)]

### Investigating the learning coefficient of modular addition: hackathon project
[[Post](https://www.lesswrong.com/posts/4v3hMuKfsGatLXPgt/investigating-the-learning-coefficient-of-modular-addition)]

### A Mathematical Framework for Transformer Circuits
[[Post](https://transformer-circuits.pub/2021/framework/index.html)]

### Understanding Addition in Transformers
[[Paper](https://arxiv.org/abs/2310.13121)]
- Observation: Phase transition for each digit in integer addition
- This paper: Model trains each digit semi-independently, hence multiple phase transitions

### Multi-Component Learning and S-Curves
[[Post](https://www.alignmentforum.org/posts/RKDQCB6smLWgs2Mhr/multi-component-learning-and-s-curves)]
-  for low rank matrix, when we increase rank, more chances ...(some probability argument) ... dissipates grokking (immediate learning)