<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <!--
  <a href="https://github.com/pytorch/fairseq/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
  -->
</p>

--------------------------------------------------------------------------------
An Auto-encoder for audio semantic communication, based on wav2vec at the platfrom of facebook's Fairseq. The Fairseq code is originally from https://github.com/pytorch/fairseq

中文描述

- 要新下载模型
- 需要新建数据集分割列表 manifest 文件
- 更改了 libri_label.py 数据库
- 注意安装fairseq 的版本，采用20201031的版本可以跑通

- 文件备注
-- Harlan_Thomas/wav2vec 文件用于修改代码并且调参数
-- semantic_commun 部分记录原先已经小修的代码
-- fairseq 已经git pull 代表最新
-- fairseq_20201031 10月31日版本代码
- ASR系统参考
-- install OpenCL

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.
We provide reference implementations of various sequence modeling papers:




# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.


# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.

# Pre-trained models and examples

<!--
We provide pre-trained models and pre-processed, binarized test sets for several tasks listed below,
as well as example training and evaluation commands.

- [Translation](examples/translation/README.md): convolutional and transformer models are available
- [Language Modeling](examples/language_model/README.md): convolutional and transformer models are available 

We also have more detailed READMEs to reproduce results from specific papers:
- [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
- [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
- [Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)](examples/quant_noise/README.md)
- [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
- [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
- [Reducing Transformer Depth on Demand with Structured Dropout (Fan et al., 2019)](examples/layerdrop/README.md)
- [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md)
- [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
- [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
- [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
- [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
- [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
- [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
- [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
- [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
- [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
- [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
- [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md) 
-->

#  Fairseq community

* Twitter: https://twitter.com/fairseq
* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

```bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
