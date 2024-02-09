# CTRL
Pytorch implementation of ["Two Trades is not Baffled: Condensing Graph via Crafting Rational Gradient Matching"](https://arxiv.org/abs/2402.04924).

![ctrl](figures/ctrl.png)


## Run the code
For example, to get the condensed graph, run the following command:

```
python train_ctrl_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1  \ 
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
@misc{zhang2024trades,
      title={Two Trades is not Baffled: Condensing Graph via Crafting Rational Gradient Matching}, 
      author={Tianle Zhang and Yuchen Zhang and Kun Wang and Kai Wang and Beining Yang and Kaipeng Zhang and Wenqi Shao and Ping Liu and Joey Tianyi Zhou and Yang You},
      year={2024},
      eprint={2402.04924},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
