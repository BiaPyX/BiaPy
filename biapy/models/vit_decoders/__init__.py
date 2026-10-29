"""
Decoders of the ViT encoder-decoder architectures. Each is a mixin that builds its layers on the model itself, so
parameter names (and checkpoints) are unchanged.

- ``unetr_decoder``: UNETR's convolutional decoder ('unetr').
- ``dpt_decoder``: DPT's reassemble/fusion decoder ('dpt').
- ``readout_decoder``: Cellpose 4's per-token readout with optional refinement ('vit_readout').
- ``output_heads``: output heads of 'dpt' and 'vit_readout'.
"""
