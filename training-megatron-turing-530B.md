# Training Megatron-Turing NLG 530B
Ref: https://arxiv.org/pdf/2201.11990

Following will be my train of though as I myself learn.

This was the then (2022) single largest monolithic LLM. The papers rightaway states the following two challenges in training such massive LLM:
A. How do you fit a monster of this size on a single card? Spoiler: You don't! That much is obvious - you somehow need to split the model among multiple GPUs. Do you use each card for a single block / layer (model parallelism) or for a single parameter tensor (tensor parallelism). How do you then account for the delays that orrurs in inter-GPU data transfer because these operations necessarily need to happen in series for inference!
B. The sheer amount of training data would render the training time years! How do you optimize that? Is it a really, really large effective batch size using model parallelism? This would require oh so many GPUs!

## Challenges
### Memory Efficieny
In mixed preci training, each parameter requries 20 bytes of memory. (2 + 4) for model, (2 + 4) for grads and (4 + 4) for adam states. Following is from the ICLR 2018 Mixed Preci paper.

<img width="625" height="328" alt="image" src="https://github.com/user-attachments/assets/dd8b06ec-2bfc-4d05-8115-31ad3512c529" />

And that is excluding activations, which, in the absence of any gradient checkpointing make up B X T X C X N X 2 (where N is number of layers, B is batch size, T is sequence length, C is hidden dim).

This adds up to ~ 26 Tb memory for a 0.5 T param model. Grad checkpointing surely helps for activation memory.

### Compute Efficieny
A smaller batch size training procedure suffers from a lower arithmetic intensity of the setup + noisy updates lead to worse performance. [See](https://arxiv.org/pdf/2310.03693) fig 5a.
There are still some [generalization issues](https://openreview.net/pdf?id=H1oyRlYgg) with choosing a larger batch size. 
