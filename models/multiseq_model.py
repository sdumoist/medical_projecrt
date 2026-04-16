"""
Complete multi-sequence MRI classification model.
Combines 3D encoder + CoPAS fusion + head.
"""
import torch
import torch.nn as nn

from models.encoders import get_encoder, MultiSeqEncoder3D
from models.fusion_copas import MultiSeqFusion, SimpleFusion
from models.heads import BinaryHead, TernaryHead


class MultiSeqClassifier(nn.Module):
    """Main classifier for multi-sequence MRI."""

    def __init__(
        self,
        num_sequences=5,        # MRI序列数量
        num_diseases=7,          # 疾病类别数
        encoder_name="resnet3d_18",  # 3D编码器类型
        num_classes=2,          # 分类类别数
        pretrained=False,      # 是否使用预训练权重
        hidden_dim=256,         # 隐藏层维度
        dropout=0.3,           # Dropout比率
        fusion_type="copas",   # 融合方式: copas/simple
        use_co_att=True,       # 是否使用共注意力
        use_cross_modal=True   # 是否使用跨模态
    ):
        super().__init__()  # 初始化nn.Module

        self.num_sequences = num_sequences
        self.num_diseases = num_diseases
        self.num_classes = num_classes
        self.encoder_name = encoder_name

        # 创建3D编码器
        self.encoder = MultiSeqEncoder3D(
            num_sequences=num_sequences,
            encoder_name=encoder_name,
            in_channels=1,
            pretrained=pretrained
        )

        feature_dim = self.encoder.feature_dim  # 获取编码器输出维度

        # 根据fusion_type选择融合模块
        if fusion_type == "copas":
            self.fusion = MultiSeqFusion(
                num_sequences=num_sequences,
                feature_dim=feature_dim,
                use_co_att=use_co_att,
                use_cross_modal=use_cross_modal,
                dropout=dropout
            )
            fused_dim = feature_dim // 2
        else:
            self.fusion = SimpleFusion(
                num_sequences=num_sequences,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            fused_dim = hidden_dim

        # 根据num_classes选择预测头
        if num_classes == 2:
            self.head = BinaryHead(fused_dim, num_diseases, dropout)
        else:
            self.head = TernaryHead(fused_dim, num_diseases, dropout)

    def forward(self, x, return_features=False):
        """Forward pass.

        Args:
            x: [B, num_seq, C, D, H, W]
            return_features: Whether to return intermediate features

        Returns:
            logits: [B, num_diseases] for binary, [B, num_diseases, 3] for ternary
            features: (optional) [B, fused_dim]
        """
        # Encode each sequence
        features = self.encoder(x)  # [B, num_seq, feature_dim]

        # Fusion
        fused = self.fusion(features)  # [B, fused_dim]

        # Predict
        logits = self.head(fused)

        if return_features:
            return logits, fused
        return logits

    def get_token(self, x):
        """Get token for downstream tasks (e.g., Qwen SFT/RL).

        Args:
            x: [B, num_seq, C, D, H, W]

        Returns:
            token: [B, hidden_dim]
        """
        _, token = self.forward(x, return_features=True)
        return token


def create_model(
    encoder="resnet3d_18",
    num_classes=2,
    num_diseases=7,
    pretrained=False,
    hidden_dim=256,
    dropout=0.3,
    fusion="copas"
):
    """Create model from config."""
    return MultiSeqClassifier(
        num_sequences=5,
        num_diseases=num_diseases,
        encoder_name=encoder,
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        dropout=dropout,
        fusion_type=fusion
    )