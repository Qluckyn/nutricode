"""
原型引导对齐模块（Proto Align）
维护每个类别在 CLIP 特征空间的原型向量，计算合成图的原型对齐损失。
只依赖 torch，不依赖项目其他模块。
"""
import torch
import torch.nn.functional as F


class PrototypeManager:
    """
    管理 CLIP 特征空间中的类原型向量。
    原型 = 真实样本特征的指数移动平均（EMA）。
    """

    def __init__(
        self,
        n_classes: int,
        feat_dim: int = 512,
        momentum: float = 0.999,
        device: str = "cuda",
    ):
        """
        Args:
            n_classes: 类别数（本项目为 2）
            feat_dim:  CLIP ViT-B/16 图像特征维度（固定 512）
            momentum:  EMA 动量，越大原型越稳定，建议 0.999
            device:    计算设备
        """
        self.n_classes = n_classes
        self.momentum = momentum
        self.device = device
        # 原型初始化为全零，第一个 batch 后被真实样本覆盖
        self.prototypes = torch.zeros(n_classes, feat_dim).to(device)
        self._initialized = [False] * n_classes

    @torch.no_grad()
    def update(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        real_mask: torch.Tensor,
    ):
        """
        用当前 batch 的真实样本特征更新原型（EMA）。

        Args:
            feats:      图像特征，shape (B, 512)，已 L2 归一化，float32
            labels:     整数标签，shape (B,)，值 0 或 1（必须是 label_origin）
            real_mask:  布尔掩码，shape (B,)，True = 真实样本
        """
        real_feats = feats[real_mask]
        real_labels = labels[real_mask]

        for c in range(self.n_classes):
            cls_mask = (real_labels == c)
            if cls_mask.sum() == 0:
                continue
            cls_mean = real_feats[cls_mask].mean(dim=0)
            cls_mean = F.normalize(cls_mean, dim=0)

            if not self._initialized[c]:
                self.prototypes[c] = cls_mean
                self._initialized[c] = True
            else:
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c]
                    + (1.0 - self.momentum) * cls_mean
                )
                self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)

    def compute_loss(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        synth_mask: torch.Tensor,
        margin: float = 0.2,
    ) -> torch.Tensor:
        """
        计算合成样本的原型对齐损失（Triplet-style）。

        对每张合成图：
            d_pos = 1 - cos(f_syn, p_y)    与本类原型的余弦距离
            d_neg = 1 - cos(f_syn, p_ȳ)   与异类原型的余弦距离
            loss_i = max(0, d_pos - d_neg + margin)

        Args:
            feats:      图像特征，shape (B, 512)，已 L2 归一化
            labels:     整数标签，shape (B,)（必须是 label_origin）
            synth_mask: 布尔掩码，shape (B,)，True = 合成样本
            margin:     triplet margin

        Returns:
            标量 loss（若无合成样本或原型未初始化则返回 0）
        """
        if not synth_mask.any():
            return torch.tensor(0.0, device=self.device)

        if not all(self._initialized):
            # 原型未完成初始化，跳过（训练开始几步内）
            return torch.tensor(0.0, device=self.device)

        # 确保 feats 是 float32（FP16 训练时可能是 float16）
        feats = feats.float()

        syn_feats = feats[synth_mask]       # (N_syn, 512)
        syn_labels = labels[synth_mask]     # (N_syn,)

        proto_pos = self.prototypes[syn_labels]       # (N_syn, 512)
        proto_neg = self.prototypes[1 - syn_labels]   # (N_syn, 512)，只有两类才成立

        # feats 和 prototypes 均已归一化，点积 = 余弦相似度
        sim_pos = (syn_feats * proto_pos).sum(dim=1)   # (N_syn,)
        sim_neg = (syn_feats * proto_neg).sum(dim=1)   # (N_syn,)

        d_pos = 1.0 - sim_pos
        d_neg = 1.0 - sim_neg

        loss = F.relu(d_pos - d_neg + margin)
        return loss.mean()


if __name__ == "__main__":
    """独立单元测试，运行：python proto_align.py"""
    pm = PrototypeManager(n_classes=2, feat_dim=512, momentum=0.999)

    B = 8
    feats = torch.randn(B, 512).cuda()
    feats = F.normalize(feats, dim=1)
    feats.requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]).cuda()
    real_mask = torch.tensor([True, True, True, True,
                              False, False, False, False]).cuda()
    synth_mask = ~real_mask

    pm.update(feats, labels, real_mask)

    assert all(pm._initialized), "原型应已初始化"
    assert abs(pm.prototypes[0].norm().item() - 1.0) < 1e-4, "原型0未归一化"
    assert abs(pm.prototypes[1].norm().item() - 1.0) < 1e-4, "原型1未归一化"
    print(f"prototypes initialized: {pm._initialized}")
    print(f"prototype[0] norm: {pm.prototypes[0].norm().item():.6f}")
    print(f"prototype[1] norm: {pm.prototypes[1].norm().item():.6f}")

    loss = pm.compute_loss(feats, labels, synth_mask, margin=0.2)
    print(f"proto loss: {loss.item():.6f}")
    assert loss.item() >= 0, "loss 应为非负数"
    loss.backward()
    print("backward OK")
    print("所有断言通过 ✓")
