import torch.nn as nn
from paper_ablation_MTI_HANet import MTI_HANet_Ours


class MTI_HANet(MTI_HANet_Ours):
    """
    兼容层：保持旧训练脚本 `from gemini_MTI_HANet import MTI_HANet` 可用。
    结构复用 paper 中的完整体（α + M），并提供差分学习率分组接口。
    """

    def get_optimizer_groups(self, base_lr=1e-5, new_module_lr_multiplier=10.0):
        backbone_params = []
        new_module_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "cnnbackbone" in name:
                backbone_params.append(param)
            else:
                new_module_params.append(param)
        return [
            {"params": backbone_params, "lr": base_lr},
            {"params": new_module_params, "lr": base_lr * new_module_lr_multiplier},
        ]

