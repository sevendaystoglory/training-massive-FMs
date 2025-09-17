# Training Megatron-Turing NLG 530B
Ref: https://arxiv.org/pdf/2201.11990

Following will be my train of though as I myself learn.

This was the then (2022) single largest monolithic LLM. The papers rightaway states the following two challenges in training such massive LLM:
A. How do you fit a monster of this size on a single card? Spoiler: You don't! That much is obvious - you somehow need to split the model among multiple GPUs. Do you use each card for a single block / layer (model parallelism) or for a single parameter tensor (tensor parallelism). How do you then account for the delays that orrurs in inter-GPU data transfer because these operations necessarily need to happen in series for inference!
B. The sheer amount of training data would render the training time years! How do you optimize that? Is it a really, really large effective batch size using model parallelism? This would require oh so many GPUs!

## Challenges
### Memory Efficieny
In mixed preci training, each parameter requries 20 bytes of memory. (2 + 4) for model, (2 + 4) for grads and (4 + 4) for adam states. 
<img width="625" height="328" alt="image" src="https://github.com/user-attachments/assets/dd8b06ec-2bfc-4d05-8115-31ad3512c529" /> [^1]
[^1] : From the ICLR 2018 Mixed Preci paper.
