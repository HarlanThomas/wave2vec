# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .. import FairseqDataset


logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        # raw data processed added on 2nd Feb 2021 by thn
        self.feature_dim = 32
        self.time_compress_rete = 160

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        # when raw audio is smaller than target_size
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end], start

    

    def crop_to_max_size_decoder(self, wav, target_size, start_of_eb):
        # according to embedding's starting, cut raw audio data
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav
        
        start = start_of_eb # // 512
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]
        label_sources = [s["label"] for s in samples]
        print()
        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)
        # print("*****Target_S", target_size)    # commented on 0302
        
        collated_sources = sources[0].new_zeros(len(sources), target_size)
        label_collated_sources = label_sources[0].new_zeros(len(label_sources), target_size) #  //512*160)

        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size, label_sources) in enumerate(zip(sources, sizes, label_sources)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
                collated_sources[i] = label_sources
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                label_collated_sources[i] = torch.cat(
                    [label_sources, label_sources.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i], start_eb = self.crop_to_max_size(source, target_size)
                # print("label_sources",label_sources.shape, "target_size", target_size ) # 0302 commented
                # label_collated_sources[i] = self.crop_to_max_size_decoder(label_sources, target_size//512*160, start_eb)
                label_collated_sources[i] = self.crop_to_max_size_decoder(label_sources, target_size, start_eb)
        input = {"source": collated_sources}
        label = {"label_source": label_collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask

        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input, "label": label}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )

        self.fnames = []
        self.label_path = '/home/thn/audio_dataset/LibriSpeech/thn_train_clean/'
        skipped = 0
        print('manifest_path:',manifest_path)
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                print('\n', items)
                assert len(items) == 2, line
                sz = int(items[1])
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")
        logger.info(f"{self.fnames}")
    def __getitem__(self, index):
        import soundfile as sf
        
        fname = os.path.join(self.root_dir, self.fnames[index])
        # print('***fname', fname)
        if self.fnames[index].split('.')[-1] == 'flac':
            wav, curr_sample_rate = sf.read(fname)
            feats = torch.from_numpy(wav).float()
            feats = self.postprocess(feats, curr_sample_rate)
            # print("**len(wav)", len(wav), "***|feats|", feats.shape)
        else:
            import numpy as np
            embed = np.loadtxt(fname)
            feats = torch.from_numpy(embed).float()
            # print("***|feats_S|", feats.shape)

        
        l_fname = os.path.join(self.label_path, self.fnames[index].split('.')[0] + '.flac')
        label_wav, curr_sample_rate = sf.read(l_fname)
        label_feats = torch.from_numpy(label_wav).float()
        label_feats = self.postprocess(label_feats, curr_sample_rate)
        # logger.info(f" ****|source_audio_file_sz|:{feats.shape}")
        # print('\n ****|source_audio_file_S|:', feats.shape)

        return {"id": index, "source": feats, "label": label_feats}
