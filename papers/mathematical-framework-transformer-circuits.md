---
date: 2021-12-22
time: 15:54
author: 
title: A Mathematical Framework for Transformer Circuits
created-date: 2025-01-15
tags: 
paper: https://transformer-circuits.pub/2021/framework/index.html
code: 
zks-type: lit
---
- "mechanistic interpretability, attempting to reverse engineer the detailed computations performed by transformers"

---
## Description of result
 reverse engineer several toy, attention-only models
- **Zero layer transformers model bigram statistics**. The bigram table can be accessed directly from the weights.
- **One layer attention-only transformers are an ensemble of bigram and “skip-trigram” (sequences of the form "A… B C") models.** The bigram and skip-trigram tables can be accessed directly from the weights, without running the model. .. implementing a kind of very simple in-context learning.
- **Two layer attention-only transformers can implement much more complex algorithms using compositions of attention heads.** These compositional algorithms can also be detected directly from the weights. Notably, two layer models use attention head composition to create “induction heads”, a very general in-context learning algorithm (see [icl-induction-heads](icl-induction-heads.md))
- One layer and two layer attention-only transformers use **very different algorithms** to perform in-context learning. Two layer attention heads induction heads

---
# How it compares to previous work
- Distill circuits [thread](https://distill.pub/2020/circuits/) for vision models but none for transformers / LMs.

---
# Main strategies used to obtain results
foo

