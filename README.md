# FlashAttention — Derivation and Implementation Notes

This repository contains a concise mathematical and implementation-oriented explanation of **FlashAttention**, an IO-aware algorithm for computing *exact* self-attention efficiently by reducing memory traffic. Please refer to report in this repository.

The report covers:

- Why standard self-attention is memory-bound  
- Numerically stable softmax and the log-sum-exp trick  
- Online softmax and tiling strategies  
- Full forward pass of FlashAttention  
- Complete backward-pass derivation with index-level equations  
- Blockwise backward algorithm (FlashAttention v1 style)  
- How these concepts map to Triton GPU kernels  
 

## References
- **FlashAttention Paper** — T. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135  
- **YouTube** — Umar Jamil, *Flash Attention derived and coded from first principles with Triton*  
- **FlashAttention GitHub** — https://github.com/Dao-AILab/flash-attention  
- **Triton Documentation** — https://triton-lang.org  

---