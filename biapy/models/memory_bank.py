"""
This module implements a Memory Bank for contrastive learning, designed to store and manage queues of pixel-level and segment-level features.

The `MemoryBank` class facilitates the use of a feature queue, which is a common
component in self-supervised contrastive learning methods. It allows for the
dynamic updating of stored features, ensuring that the bank always contains
a diverse and up-to-date set of representations for each class. This is crucial
for contrastive losses that rely on a large number of negative samples.
"""
import torch
import torch.nn as nn
from typing import Tuple


class MemoryBank(nn.Module):
    """
    Memory Bank for storing pixel and segment features. Used in contrastive learning to maintain a queue of features.

    Parameters
    ----------
    num_classes : int
        Number of classes in the dataset.

    memory_size : int
        Size of the memory bank for each class.

    feature_dims : int
        Dimension of the feature vectors stored in the memory bank.

    network_stride : int
        Stride of the network, used to downsample the features.

    pixel_update_freq : int
        Frequency at which pixel features are updated in the memory bank.

    device : torch.device
        Device on which the memory bank is stored (CPU or GPU).

    ignore_index : int, optional
            Value to ignore in the loss calculation. If not provided, no value will be ignored.
    """

    def __init__(
        self,
        num_classes: int = 2,
        memory_size: int = 5000,
        feature_dims: int = 256,
        network_stride: Tuple[int, ...] = (16, 16),
        pixel_update_freq: int = 10,
        device: torch.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda"),
        ignore_index: int = -1,
    ):
        """
        Initialize the MemoryBank.

        Sets up two main queues: `pixel_queue` for storing individual pixel features
        and `segment_queue` for storing aggregated segment-level features. Each queue
        is initialized with random normalized features and includes a pointer to
        manage enqueue/dequeue operations.

        Parameters
        ----------
        num_classes : int, optional
            The total number of classes in the dataset. Each class will have its
            own dedicated queue within the memory bank. Defaults to 2.
        memory_size : int, optional
            The maximum number of feature vectors to store for each class in both
            the pixel and segment queues. Defaults to 5000.
        feature_dims : int, optional
            The dimensionality of the feature vectors that will be stored. This
            should match the output dimension of the feature extractor. Defaults to 256.
        network_stride : Tuple[int, ...], optional
            The spatial stride of the network's output features relative to the
            input image. Used to correctly downsample ground truth labels to match
            feature map dimensions. For 2D, e.g., (16, 16); for 3D, e.g., (8, 16, 16).
            Defaults to (16, 16).
        pixel_update_freq : int, optional
            The maximum number of pixel features to enqueue into the `pixel_queue`
            for a given class in a single update step. This helps control the
            update rate and memory usage. Defaults to 10.
        device : torch.device, optional
            The PyTorch device (e.g., `torch.device('cuda')` or `torch.device('cpu')`)
            on which the memory bank tensors will be allocated and stored.
            Defaults to CUDA if available, otherwise CPU.
        ignore_index : int, optional
            A class label value that should be ignored during feature extraction
            and enqueueing (e.g., for background or unlabeled regions). Features
            associated with this label will not be added to the memory bank.
            Defaults to -1.
        """
        super(MemoryBank, self).__init__()

        # Memory bank
        self.num_classes = num_classes
        self.memory_size = memory_size
        self.feature_dims = feature_dims
        self.network_stride = network_stride
        self.pixel_update_freq = pixel_update_freq
        self.ignore_index = ignore_index

        self.pixel_queue = torch.randn(num_classes, memory_size, feature_dims).to(device)
        self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
        self.pixel_queue_ptr = torch.zeros(num_classes, dtype=torch.long).to(
            device
        )  # Pointer to track the next position to enqueue

        self.segment_queue = torch.randn(num_classes, memory_size, feature_dims).to(device)
        self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
        self.segment_queue_ptr = torch.zeros(num_classes, dtype=torch.long).to(device)

    def dequeue_and_enqueue(self, keys: torch.Tensor, labels: torch.Tensor):
        """
        Dequeue and enqueue features into the memory bank.

        Parameters
        ----------
        keys : torch.Tensor
            Features to be enqueued, shape (batch_size, classes, H, W) or (batch_size, classes, D, H, W).
            E.g. (8, 19, 128, 256) for a batch size of 2, 19 classes, and a spatial size of 128x256.

        labels : torch.Tensor
            Ground truth labels, shape (batch_size, 1, H, W) or (batch_size, 1, D, H, W).
            E.g. (8, 1, 128, 256) for a batch size of 2 and a spatial size of 128x256.
        """
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        # When working in instance segmentation the channels are more than 1 so we need to merge then into
        # just one channel. This trick of multiplying an offset is to take into account the background class too.
        if labels.shape[1] != 1:
            if labels.ndim == 4:
                offsets = torch.tensor([1, 2], device=labels.device).view(1, 2, 1, 1)
            else:
                offsets = torch.tensor([1, 2], device=labels.device).view(1, 2, 1, 1, 1)
            labels = labels * offsets
            labels, _ = labels.max(dim=1)
        # In semantic the target is already compressed into one channel
        else:
            labels = labels.squeeze(1)
        labels = labels.long()

        # Downsample the labels according to the network stride
        if labels.ndim == 3:
            labels = labels[:, :: self.network_stride[-2], :: self.network_stride[-1]]
        else:
            labels = labels[:, :: self.network_stride[-3], :: self.network_stride[-2], :: self.network_stride[-1]]

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_index and x != 0]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)

                ptr = int(self.segment_queue_ptr[lb])

                self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.memory_size:
                    self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    self.pixel_queue[lb, ptr : ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.memory_size
