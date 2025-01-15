---
date: 2025-01-15
time: 16:01
author: 
title: In-context Learning and Induction Heads
created-date: 2025-01-15
tags: 
paper: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
code: 
zks-type: lit
---
## Intro

- Induction Heads: circuit that copies and completes sequences that have occurred before. eg if the sequence is `... [A][B]......[A]`, then the induction head predicts `[B]`
- "Mechanically, induction heads in our models are implemented by a circuit of two attention heads:"
    - 'the first head is a “previous token head” which copies information from the previous token into the next token'
    - 'the second head (the actual “induction head”) uses that information to find tokens preceded by the present token.' 
    - "For 2-layer attention-only models, we were able to show precisely that induction heads implement this pattern copying behavior and appear to be the primary source of in-context learning."

- Hypothesis: "induction heads might constitute the mechanism for the actual majority of all in-context learning in large transformer models" 
    - how? show that the phase change (visible bump in training loss) early in training for language models of various size relates to the acquisition of ICL, and show that it is causal (eg intentionally causing the bump to appear at different epochs of training causes ICL ability to be acquired around that new epoch). (note: this is about general ICL, not just literal copying of tokens)
    - evidence is strong and causal for small models, but medium and correlational for large models

---

## Description of result

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

---
## How it compares to previous work
foo

---
## Main strategies used to obtain results
- Per-Token Loss Analysis allows us to analyze and compare training trajectories
    - run model / snapshot over a set of examples, collecting one token's loss per example
    - for each sample, extract loss of a consistent token and combine them into a vector
    - analyze vector by PCA and 2D projection
- Heuristic measure of ICL ("ICL score"): loss of 500th token - avg loss of 50th token in the context, averaged over dataset examples. (authors also varied the numbers to show that conclusions do not change)

---
## Other
### Unexplained curiosities
- Seemingly constant ICL score after phase change, across all models
- The ordering of derivative of loss wrt log(train tokens) for models of various sizes invert at phase change! (originally small models improve faster, but after phase change, large models improve faster)
### Additional Curiosities
- 6-layer attention-only model has a head that is not an induction head but ablating it has similar effect to reversing the phase change
- "4-layer MLP model ablations are nowhere near as “peaky” as those of any other model"
- "6-layer MLP model shows a “loss spike”"
-  "6-layer MLP model has one lone induction head whose ablation has the opposite effect on the in-context learning score"
- "Full-scale models above 16 layers start to show a small number of heads that score well on “prefix search”, but get a negative score on copying, which means they are not induction heads"
### Discussion
- studying ICL is impt for safety as "model behavior can in some sense “change” during inference, without further training", and this behavior can be unwanted
- "We did not observe any evidence of mesa-optimizers."
### Other
- "Circuits thread tried to extend this notion of universality from features to circuits, finding that not only do at least some families of well-characterized neurons reoccur across multiple networks of different architectures and that the same circuits, but the same circuits appear to implement them"
