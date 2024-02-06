# CTRL
Pytorch implementation of "Two Trades is not Baffled: Condense Graph via Crafting Rational Gradient Matching"

![ctrl](figures/ctrl.png)


## Run the code
For example, to get the condensed graph, run the following command:

```
python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1  \ 
--lr_adj=0.1  --r=0.001 --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=0 --alpha=0 --beta=0.2 --init_way=K-means
```

## Requirements
Please see [environment](/environment.yaml).

Run the following command to install:
```
conda env create -f environment.yaml
```

## Acknowledgement
Our code is built upon [GCond](https://github.com/ChandlerBang/GCond) 

## Cite 
If you find this repo to be useful, please cite our paper. 

```
@inproceedings{zhang2024Navigating,
  title={Navigating Complexity: Toward Lossless Graph Condensation via Expanding Window Matching},
  author={Yuchen Zhang and Tianle Zhang and Kai Wang and Ziyao Guo and Yuxuan Liang and Xavier Bresson and Wei Jin and Yang You},
  journal={arXiv preprint arXiv:2402},
  year={2024}
  }
```
