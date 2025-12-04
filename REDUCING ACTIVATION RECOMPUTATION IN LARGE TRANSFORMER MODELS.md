# REDUCING ACTIVATION RECOMPUTATION IN LARGE TRANSFORMER MODELS

<img width="962" height="309" alt="image" src="https://github.com/user-attachments/assets/6e73fce8-c75d-41d6-8b12-348fd8805baa" />

This paper, among other things, does a very good job of calculating the massive memory footprint that intermediate activations have in training massive transformers. Assuming fp16 training, we get an estimate of $sbh(34+\frac{5as}{h})$, where s is sequence length, b is mini batch size, h is hidden dimension, and a is the number of attention heads. (see page 4 of paper)
