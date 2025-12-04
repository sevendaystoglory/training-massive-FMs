# Training Megatron-Turing NLG 530B
Ref: https://arxiv.org/pdf/2201.11990

I put out my train of thought as I learn.

This was the then (2022) single largest monolithic LLM. The papers rightaway states the following two challenges in training such massive LLM:
A. How do you fit a monster of this size on a single card? Spoiler: You don't! That much is obvious - you somehow need to split the model among multiple GPUs. Do you use each card for a single block / layer (model parallelism) or for a single parameter tensor ([pipeline parallelism](https://arxiv.org/pdf/1806.03377)). How do you then account for the delays that orrurs in inter-GPU data transfer because these operations necessarily need to happen in series for inference!
B. The sheer amount of training data would render the training time years! How do you optimize that? Is it a really, really large effective batch size using model parallelism? This would require oh so many GPUs!

## Challenges
### Memory Efficiency
In mixed preci training, each parameter requries 20 bytes of memory. (2 + 4) for model, (2 + 4) for grads and (4 + 4) for adam states. Following is from the ICLR 2018 Mixed Preci paper.

<img width="625" height="328" alt="image" src="https://github.com/user-attachments/assets/dd8b06ec-2bfc-4d05-8115-31ad3512c529" />

And that is excluding activations, which, in the absence of any gradient checkpointing make up B X T X C X N X 2 (where N is number of layers, B is batch size, T is sequence length, C is hidden dim).

This adds up to ~ 26 Tb memory for a 0.5 T param model. Grad checkpointing surely helps for activation memory.

### Compute Efficieny
A smaller batch size training procedure suffers from a lower arithmetic intensity of the setup + noisy updates lead to worse performance (altough this effect operates at exceedingly low effective batch sizes). [See](https://arxiv.org/pdf/2310.03693) fig 5a.
There are still some [generalization issues](https://openreview.net/pdf?id=H1oyRlYgg) with choosing a larger batch size, that result from the model converging to a sharp minimizer. 

<img width="439" height="644" alt="image" src="https://github.com/user-attachments/assets/931e3728-ce37-451f-aaae-1d10fb6ac5b8" />


## Proposed Software: 3D Paralleism
Combines tensor, pipeline and data parallelism. Tensor parallel components have the highest memory bandwidth overhead, as each sequential unit has to wait for the previous one before proceeding to any form of computation. Hence, these parameters are placed within a node of 8xA100. Each stage of the pipeline is distributed across 35 nodes of A100, these may very well constitute distinct transformer blocks. Now, the 3D parallel further distributes the model across more such 35 node units for additional data parallelism. 


---

# Related Work
### [Measuring the Effects of Data Parallelism on Neural Network Training – JMLR, 2019](https://arxiv.org/pdf/1811.03600)
I'll briefly summarize the main contirbutions of this paper:
1. Increasing the batch size (BS) proportionally reduces the steps required to produce an out-of-sample error (test error) withing certain bounds. However, this does not hold true for very large BS, as you'd need more steps. A trainin run of BS = 2 at 40k steps may produce similar out-of-sample error as a trianing run with BS = 4 with 20k steps, but intuitively we can see that a BS = 80k would surely not converge in a single step!
2. Maximum useful BS = f(workload, properties of model, training algo, dataset).
   2.a. SGD + momentum can tolerate much higher BS than SGD.
   2.b. some models allow for much larger BS.
   2.c. effect of dataset on maximum useful BS is much lower than the effect of model and trianing algorithm.
3. Popular LR heuristics such as linearly scaling LR with BS does not hold across all BS or models.
4. Answer the 1M$ question - does increasing BS adversely affect trained model quality? They show no such evidence is found. Yet certain regularization techniques become imperative at high BS. This, atleast, partially reconciles the conflicting stances in the then-prevalent literature that increasing BS degrades model quality. 
