"""
ViT backbones shared by 'vit', 'unetr', 'dpt', 'vit_readout' and 'mae'.

- ``common``: patch embedding, predefined backbones ('MODEL.VIT_MODEL'), block construction, checks and pretrained
  weight loading.
- ``encoder``: ViT encoder mixin of 'unetr', 'dpt' and 'vit_readout'.
- ``sam3``, ``dinov3``, ``celldino``, ``cellpose_sam``: pretrained backbones (blocks and weight loaders).
- ``weights``: helpers shared by the weight loaders.
"""
