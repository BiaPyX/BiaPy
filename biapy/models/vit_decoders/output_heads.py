"""Output heads of the ViT-based dense-prediction architectures ('dpt' and 'vit_readout')."""

from typing import Dict, List

import torch
import torch.nn as nn

from biapy.models.blocks import prepare_activation_layers


class OutputHeadsMixin:
    """One 1x1 conv per output head and the packing of their outputs."""

    def _build_heads(
        self,
        in_channels: int,
        output_channels: List[int],
        output_channel_info: List[str],
        explicit_activations: bool,
        head_activations: List[str],
        return_one_tensor: bool,
    ):
        """Build the output heads."""
        if len(output_channels) == 0:
            raise ValueError("'output_channels' needs to has at least one value")
        print("Selected output channels:")
        for i, info in enumerate(output_channel_info):
            print(f"  - {i} channel for {info} output")
        self.output_channels = output_channels
        self.output_channel_info = output_channel_info
        self.return_class = "class" in output_channel_info
        self.explicit_activations = explicit_activations
        self.return_one_tensor = return_one_tensor
        if self.explicit_activations:
            assert len(head_activations) == sum(output_channels), (
                "If 'explicit_activations' is True, 'head_activations' needs to have the same number of values as "
                "'output_channels'"
            )
            self.head_activations, self.class_head_activations = prepare_activation_layers(
                head_activations, output_channel_info, output_channels
            )
            if self.return_class and self.class_head_activations is None:
                raise ValueError("If 'return_class' is True, 'head_activations' must be provided.")
        self.heads = nn.Sequential()
        for out_ch in output_channels:
            self.heads.append(nn.Conv2d(in_channels, out_ch, kernel_size=1, padding="same"))

    def _format_output(self, feats: torch.Tensor) -> Dict | torch.Tensor:
        """Apply the heads and pack the outputs as the rest of BiaPy's models."""
        class_outs, outs = [], []
        for i, head in enumerate(self.heads):
            if "class" not in self.output_channel_info[i]:
                outs.append(head(feats))
            else:
                class_outs.append(head(feats))
        outs = torch.cat(outs, dim=1)

        if self.explicit_activations:
            if len(self.head_activations) == 1:
                outs = self.head_activations[0](outs)
            else:
                for i, act in enumerate(self.head_activations):
                    outs[:, i : i + 1] = act(outs[:, i : i + 1])
            if self.return_class and self.class_head_activations is not None:
                for i, act in enumerate(self.class_head_activations):
                    class_outs[i] = act(class_outs[i])

        if not self.return_class:
            return outs
        if self.return_one_tensor:
            return torch.cat((outs, torch.argmax(torch.cat(class_outs, dim=1), dim=1).unsqueeze(1)), dim=1)
        return {"pred": outs, "class": torch.cat(class_outs, dim=1)}
