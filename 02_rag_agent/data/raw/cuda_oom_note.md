# CUDA OOM 处理笔记

训练日志中如果出现 `RuntimeError: CUDA out of memory`，通常说明当前 batch size、输入分辨率或模型显存占用超过了 GPU 可用显存。

常见处理方式：

- 降低 batch size。
- 减小输入图像分辨率。
- 开启梯度累积，在较小 batch size 下模拟较大 batch。
- 检查是否有未释放的 tensor 或不必要的缓存。
