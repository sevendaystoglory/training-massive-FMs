# How is this InternVideo2 MLLM pre-trained?

## How do we pre-train LLMs, generally speaking?
BERT-like econder-only models are pre=trained using masked language modelling (MLM) as the training objective. The encoder learns to assign bi-directional context-rich representations to the masked tokens. On the other hand, GPT-like decoder-only models are pre-trained using next token prediction (TNP) as the objetive with a C.E. loss calculated using teacher forcing. Both of these training tasks are really powerful to assign context-rich representations to tokens and to assign the next most-probable token representation correspondingly. We carry forward this analogy in the learning of spatio-temporal representations of multi-modal models. 
<img width="1305" height="377" alt="image" src="https://github.com/user-attachments/assets/65479e7c-721a-4154-8eb8-75f8e765e4f0" />


Introducing the InternVideo2 training pipeline. Which operates in 3 stages:

1 . **Stage1: Reconstructing Unmasked Video Tokens.** As the paper mentions, the basic idea is to mask about 80% of video tokens and having an encoder assign representations to the UNMASKED tokens. If trained properly, the encoder should assign similar representations in with 80% of its spatio-temporal context masked, as it would it the context were entirely unmasked. Training is, again, done using teacher forcing. I'll state the key sentence verbatim. _We only align the unmasked tokens, by minimizing their mean squared error (MSE) between student and teachers. The learning objective is to reconstruct the remaining tokens as:_

$$
L = \frac{1}{Z} \sum^p \left( \alpha_1 |f(V_p) - g(V_p)|^2 + \alpha_2 |f(V_p) - h(V_p)|^2 \right);
$$

where $f$ is our video encoder, and g and h are the teacher encoder ViTs. Specifically, InternViT-6B and ViT-g of VideoMAEv2. $V_p$ is the $p^{th}$ video token. $Z$ is a normalization factor. 

After stage 1, quite a strong video encoder is trained. However, this is still not a multi-model model yet. We need to encode text, speech and audio somehow. Not only that, we also need to ensure that these learned representations across modalities are aligned! What this means is that a video and its caption should be encoded quite closely using a good pre-trained InternVideo2.

2. **Stage2: Aligning Video to Audio-Speech-Text.** How we do this is done is that there already are good text/audio/speech encoders. We take the the BERT-L (19 encoder layers and 5 cross-attn decoder layers) for text and speech encoding. The audio encoder is initialized with BEATs. Good! What remains now is to align these cross-modal representations. This is done in 3 ways: A. using a contrastive loss to learn similar M-text embeddings, where $M\in\{video, audio, speech\}$. Call this $L_{CON}$; B. computing a loss $L_{MAT}$ which captures the likelihood of matching a given video $V$ with a caption $T$; C. Up until now the encoder layers of BERT-L were trained. We now unleash the cross-attn decoder of BERT. Given a video, and some of its caption, we train the BERT-L to complete the caption. This is autoregressive LM task. I don't know why they've called it masked language modelling 🤷. With this the final piece of loss is $L_{MLM}$. Henceforth, stage 2 loss is

$$
L = L_{CON} + L_{MAT} + L_{MLM}
$$

