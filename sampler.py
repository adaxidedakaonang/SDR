import torch
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import sys
# sys.path.append('..')
# sys.path.append('.')
# sys.path.append('...')
import argparser
from dataset.utils import Replayset
from utils.run_utils import *
import math

class InterleaveSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.datasets_num = len(self.dataset.datasets)
        self.batch_size = batch_size
        self.largest_dataset_size = max([len(d) for d in self.dataset.datasets])
        print()

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.datasets_num):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)
        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.datasets_num // 2
        samples_to_grab = self.batch_size//2
        epoch_samples = self.largest_dataset_size * self.datasets_num
        
        final_sample_list = []
        for _ in range(0,epoch_samples, step):
            for i in range(self.datasets_num):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for __ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # print(i)
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_sample_list.extend(cur_samples)

        return iter(final_sample_list)

    def __len__(self):
        return math.ceil(self.largest_dataset_size/self.batch_size) * self.datasets_num


if __name__ == "__main__":
    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)
    opts.data_root = r"D:\\ADAXI\\Datasets\\VOC_SDR"
    opts.replay = True
    opts.mix = True
    opts.task = '10-10'
    opts.dataset = 'voc'
    opts.step = 1
    train_dst, val_dst, test_dst, n_classes = get_dataset(opts, rank=0)
    batch_size = 8
    concate_dataset = ConcatDataset([train_dst.dataset, train_dst.replayset])
    interleave_sampler = InterleaveSampler(concate_dataset, batch_size=8)
    loader = DataLoader(concate_dataset, sampler=interleave_sampler, batch_size=1)
    print()
    for _, (image, label) in enumerate(loader):
        # print(image)
        # print(label)
        print(_)
    print("good")