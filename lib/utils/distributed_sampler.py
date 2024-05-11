import torch
import random
import math
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    `torch.utils.data.DistributedSampler`.

    In pytorch of lower versions, there is no `shuffle` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 batch_size=4):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0
        self.batch_size = batch_size if batch_size is not None else 1
        self.num_batches_per_dataset = [math.ceil(len(ds) / batch_size) for ds in self.dataset.datasets]
        self.dataset_lengths = [len(ds) for ds in self.dataset.datasets]
        self.dataset_start_idx = [0]
        for ds in self.dataset.datasets[:-1]:
            self.dataset_start_idx.append(self.dataset_start_idx[-1] + len(ds))

    def __iter__(self):
        """Deterministically shuffle based on epoch."""

        # if self.shuffle:
        #     g = torch.Generator()
        #     g.manual_seed(self.epoch + self.seed)
        #     indices = torch.randperm(len(self.dataset), generator=g).tolist()
        # else:
        #     indices = torch.arange(len(self.dataset)).tolist()
        #
        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size
        #
        # # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        # assert len(indices) == self.num_samples
        # return iter(indices)
        ''' intra-batch is same, and inter-batch is different'''
        # batch_order = []
        # # 对每个数据集生成 batch 索引
        # for dataset_idx, num_batches in enumerate(self.num_batches_per_dataset):
        #     for batch_idx in range(num_batches):
        #         batch_order.append((dataset_idx, batch_idx))
        # # 打乱 batch 顺序，以确保随机选择数据集
        # if self.shuffle:
        #     g = torch.Generator()
        #     g.manual_seed(self.epoch + self.seed)
        #     order_indices = torch.randperm(len(batch_order), generator=g).tolist()
        #     batch_order = [batch_order[i] for i in order_indices]
        #
        # dataset_indexes = []
        # for dataset_idx, batch_idx in batch_order:
        #     base_idx = dataset_idx if dataset_idx == 0 else self.dataset.cumulative_sizes[dataset_idx - 1]
        #     start_idx = batch_idx * self.batch_size + base_idx
        #     end_idx = start_idx + self.batch_size
        #     padding_idx = end_idx - self.dataset.cumulative_sizes[dataset_idx]
        #     dataset_index = list(range(start_idx, end_idx))
        #     if padding_idx > 0:
        #         dataset_index = list(range(start_idx, self.dataset.cumulative_sizes[dataset_idx])) + list(range(base_idx, base_idx + padding_idx))
        #     dataset_indexes += dataset_index

        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        datasets_indices = [(torch.randperm(length, generator=g) + self.dataset_start_idx[i]).tolist() for i, length in enumerate(self.dataset_lengths)]
        batches = []
        for dataset_idx, indices in enumerate(datasets_indices):
            if not self.drop_last:
                num_full_batches, leftover = divmod(len(indices), self.batch_size)
                # attach full batch
                for batch_idx in range(num_full_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = start_idx + self.batch_size
                    batches.append(indices[start_idx:end_idx])
                if leftover:
                    batches.append(indices[-leftover:])
            else:
                for batch_idx in range(len(indices) // self.batch_size):
                    start_idx = batch_idx * self.batch_size
                    end_idx = start_idx + self.batch_size
                    batches.append(indices[start_idx:end_idx])
        random.shuffle(batches)
        random.shuffle(batches)
        batches = [item for sublist in batches for item in sublist]
        return iter(batches)
