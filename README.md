# metaCDA
metaCDA: A Novel Framework for CircRNA-Driven Drug Discovery Utilizing Adaptive Aggregation and MetaKnowledge Learning

### Torch version is available now!

This repository contains pyTorch code and datasets for the paper.

>Li Peng, Huaping Li, Sisi Yuan, Tao Meng, Yifan Chen∗, Xiangzheng Fu，and Dongsheng Cao∗

## Introduction
In the emerging field of RNA drugs, circRNA has attracted much attention as a novel multi-functional therapeutic target. Delve deeper into the intricate interactions between circRNA and disease is critical for driving drug discovery efforts centered around circRNAs. Current computational methods face two significant limitations: lack of aggregate information in heterogeneous graph networks and lack of higher-order fusion information. To this end, we present a novel approach, metaCDA, which utilizes metaknowledge and adaptive aggregate learning to improve the accuracy of circRNA and disease association predictions and addresses both limitations. We calculate multiple similarity measures between disease and circRNA, and construct heterogeneous graph based on these, and apply meta-networks to extract meta-knowledge from the heterogeneous graph, so that the constructed heterogeneous maps have adaptive contrast enhancement information. Then, we construct a nodal adaptive attention aggregation system, which integrates multi-head attention mechanism and nodal adaptive attention aggregation mechanism, so as to achieve accurate capture of higher-order fusion information. We conducted extensive experiments, and the results show that metaCDA outperforms existing state-of-the-art models and can effectively predict disease associated circRNA, opening up new prospects for circRNA-driven drug discovery.

## Citation
If you want to use our codes and datasets in your research, please cite:
under maintenance...

## Environment
The codes of metaCDA are implemented and tested under the following development environment:

pyTorch:
* python=3.9.16
* torch=1.12.0
* numpy=1.24.3
* pandas=1.5.3
* scipy=1.10.1
* dgl=1.1.0
* scikit-learn=1.2.0

## DATASET
Please use the link to download:：https://pan.baidu.com/s/1FrmIEji-NsDm7cyy7Jh0Sw?pwd=5exq code：5exq

Please send a email to our team to get the extraction code：1873288525@qq.com
