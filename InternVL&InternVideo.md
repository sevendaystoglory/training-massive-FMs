## Loading and printing the AutoModelForImageTexttoText object for OpenGVLab/InternVL3_5-2B-HF.

There is an image encoder. Each image is broken down into patches of 14x14 pixels that do not overlap. With three channels, each patch i.e., 3x14x14 tensor is a token. Yes, tokens are continuous here. After embedding using a CNN (and not a lookup table), the representations go through a vision transformer, and are finally, through a MultiModalProjector, mapped to the Qwen3 space. Now, any text input by the user is also embedded using the Qwen3 embedding table and appeneded to the image representations after the MultiModalProjector. This is now effectively LLM input and everything proceeds as in any causal LLM. 

```
InternVLForConditionalGeneration(
  (model): InternVLModel(
    (vision_tower): InternVLVisionModel(
      (embeddings): InternVLVisionEmbeddings(
        (patch_embeddings): InternVLVisionPatchEmbeddings(
          (projection): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
        )
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (encoder): InternVLVisionEncoder(
        (layer): ModuleList(
          (0-23): 24 x InternVLVisionLayer(
            (attention): InternVLVisionAttention(
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (projection_layer): Linear(in_features=1024, out_features=1024, bias=True)
              (projection_dropout): Identity()
              (q_norm): Identity()
              (k_norm): Identity()
            )
            (mlp): InternVLVisionMLP(
              (activation_fn): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layernorm_before): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (layernorm_after): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (layernorm): Identity()
    )
    (multi_modal_projector): InternVLMultiModalProjector(
      (layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
      (linear_1): Linear(in_features=4096, out_features=2048, bias=True)
      (act): GELUActivation()
      (linear_2): Linear(in_features=2048, out_features=2048, bias=True)
    )
    (language_model): Qwen3Model(
      (embed_tokens): Embedding(151936, 2048)
      (layers): ModuleList(
        (0-27): 28 x Qwen3DecoderLayer(
          (self_attn): Qwen3Attention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2048, out_features=1024, bias=False)
            (v_proj): Linear(in_features=2048, out_features=1024, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
            (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
          )
          (mlp): Qwen3MLP(
            (gate_proj): Linear(in_features=2048, out_features=6144, bias=False)
            (up_proj): Linear(in_features=2048, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=2048, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen3RMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen3RMSNorm((2048,), eps=1e-06)
        )
      )
      (norm): Qwen3RMSNorm((2048,), eps=1e-06)
      (rotary_emb): Qwen3RotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
```

## Understanding the architecture of OpenGVLab/InternVideo2-Stage2_6B.
The general architecture is quite similar to the InternVL model, however the InternVideo diverges as follows.
1. Its primary encoder is a large spatio-temporal ViT video embeddings, not just a simple ViT.
2. To process videos, InternVideo also has add positional embeddings.
3. InternVL recipe for image context text generation is to use Qwen3, a decoder only model with visual representations prepended to textual representations after being mapped ot the same space. InternVideo uses BERT-style model with cross-attn fusion of visual and textual embeddings.

The following figure from this [survey](https://arxiv.org/pdf/2306.13549), shows how the general architecture matches. 

<img width="509" height="416" alt="image" src="https://github.com/user-attachments/assets/64326c8b-fa61-4fff-9f6c-7d54e2a8e510" />

