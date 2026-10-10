# Resource and parameter estimate

For `google/gemma-2-2b-it`, the Hugging Face model metadata reports
2,614,341,888 frozen BF16 parameters (about 4.87 GiB of weights). Under the
explicit direct-parallel-head interpretation used here, the trainable modules
contain 194,620,953 parameters: 660,480 in the shared projection, 132,096 in the
compiler trunk, 66,304 in the key head, 193,691,648 in factor heads, 4,112 in the
mask head, and 66,313 in routing outside frozen BGE.

Trainable parameters occupy about 742.42 MiB in FP32. A simple AdamW accounting
for FP32 parameters, gradients, and two moments is about 2.90 GiB. This excludes
frozen BGE weights, activations, attention workspaces, teacher/student forwards,
token batches, and allocator overhead. Therefore it is not a peak-memory promise.

The paper reports 13.43 GB additional compile memory for Doc-to-Atom/Gemma and
24.02 GB for its D2L comparison in Table 3; it does not publish an end-to-end
training peak. The exact paper runtime is 8 x A100 80GB. For a single-GPU Molab
run, start with the resource-constrained configuration and expect a 24 GB card to
be only an experimental lower bound; reduce batch/sequence length or use ZeRO-2
if it does not fit. The direct compiler-head factorization is unpublished, so the
official trainable count may differ substantially.
