# RSyn_Net Model Names

模型文件按网络结构和关键机制命名。训练和测试入口只使用下表中的新模型名。

| Model name | Feature summary |
| --- | --- |
| `attention_residual_cnn` | Residual CNN blocks with block self-attention. |
| `residual_dncnn` | Standard DnCNN-style Conv-BN-ReLU stack with residual output `x - F(x)`. |
| `groupnorm_cnn` | GroupNorm DnCNN-style stack with direct output `F(x)`. |
| `groupnorm_residual_dncnn` | GroupNorm DnCNN-style stack with residual output `x - F(x)`. |
| `hybrid_branch_cnn` | Hybrid multi-branch CNN using ordinary convolutions and residual HN blocks. |
| `ql_hybrid_branch_cnn` | Hybrid multi-branch CNN using QLConv multiplicative-additive branches. |
| `ql_residual_block_cnn` | Residual-block CNN built from QLConv layers. |
| `quadratic_residual_dncnn` | QConv DnCNN variant using an `x**2` branch and residual output `x - F(x)`. |
| `ql_direct_dncnn` | QLConv DnCNN-style stack with direct output `F(x)`. |
| `swinir_restoration` | SwinIR image restoration network based on residual Swin Transformer blocks. |
| `standard_unet` | Classic U-Net with 64 base channels and bilinear upsampling. |
| `compact_unet` | Compact configurable U-Net with 32 base channels. |
| `haar_wavelet_unet` | Haar wavelet U-Net with subband separable convolutions. |
| `haar_wavelet_unet_gn` | Haar wavelet U-Net with GroupNorm in subband blocks. |
| `haar_wavelet_subband_attention_unet` | Haar wavelet U-Net with subband attention and shared ECA refinement. |
| `residual_attention_unet` | Residual/linear-attention U-Net with weight-standardized convolutions. |
| `mask_guided_unet` | Mask-guided U-Net that concatenates input with `1 - mask`. |

Example:

```powershell
python main/train_main.py --model haar_wavelet_subband_attention_unet
```
