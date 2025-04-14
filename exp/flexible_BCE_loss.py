import torch
import torch.nn as nn

class FlexibleBCELoss(nn.Module):
    def __init__(self, pos_weight, device):
        super().__init__()
        self.raw_pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.raw_pos_weight.to(device))

    def forward(self, input, target):
        if input.device != self.criterion.pos_weight.device:
            self.criterion.pos_weight = self.raw_pos_weight.to(input.device)
        return self.criterion(input, target)