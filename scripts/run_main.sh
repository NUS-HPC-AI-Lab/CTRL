# cora

python train_ctrl_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  \
--lr_adj=1e-4 --r=0.25  --seed=1 --epoch=800 --save=0 --alpha=0 --beta=0.9 --init_way=Random

python train_ctrl_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  \
--lr_adj=1e-4 --r=0.5  --seed=1 --epoch=800 --save=0 --alpha=0 --beta=0.9 --init_way=Random

python train_ctrl_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  \
--lr_adj=1e-4 --r=1  --seed=1 --epoch=800 --save=0 --alpha=0 --beta=0.9 --init_way=Random


# citeseer

python train_ctrl_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  \
--lr_adj=1e-4 --r=0.25  --seed=1 --epoch=800 --save=0 --alpha=0.01 --beta=0.9 --init_way=Random

python train_ctrl_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  \
--lr_adj=1e-4 --r=0.5  --seed=1 --epoch=800 --save=0 --alpha=0.01 --beta=1 --init_way=Random

python train_ctrl_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  \
--lr_adj=1e-4 --r=1  --seed=1 --epoch=800 --save=0 --alpha=0.01 --beta=0.9 --init_way=Random

# ogbn-arxiv

python train_ctrl_tranduct.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  \
--lr_adj=0.01 --r=0.001  --seed=1 --inner=3  --epochs=1000  --save=0 --alpha=0.01 --beta=0.3 --init_way=K-means

python train_ctrl_tranduct.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  \
--lr_adj=0.01 --r=0.005  --seed=1 --inner=3  --epochs=1000  --save=0 --alpha=0.01 --beta=0.7 --init_way=K-means

python train_ctrl_tranduct.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  \
--lr_adj=0.01 --r=0.01  --seed=1 --inner=3  --epochs=1000  --save=0 --alpha=0.01 --beta=0.9 --init_way=K-means


# ogbn-arxiv-xrt
# Note that dataset has only had its node feature matrix replaced with that from ogbn-arxiv.

python train_ctrl_tranduct.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  \
--lr_adj=0.01 --r=0.001  --seed=1 --inner=3  --epochs=1000  --save=0 --alpha=0 --beta=0.3 --init_way=Random_real

python train_ctrl_tranduct.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  \
--lr_adj=0.01 --r=0.005  --seed=1 --inner=3  --epochs=1000  --save=0 --alpha=0.01 --beta=0.15 --init_way=Random_real

python train_ctrl_tranduct.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  \
--lr_adj=0.01 --r=0.01  --seed=1 --inner=3  --epochs=1000  --save=0 --alpha=0 --beta=0.3 --init_way=Random_real

# flickr

python train_ctrl_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005  --gpu_id=0  \ 
--lr_adj=0.005  --r=0.001 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --alpha=0 --beta=0.7 --init_way=Random

python train_ctrl_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005  --gpu_id=0  \ 
--lr_adj=0.005  --r=0.005 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --alpha=0 --beta=0.05 --init_way=Random

python train_ctrl_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005  --gpu_id=0  \ 
--lr_adj=0.005  --r=0.01 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=0 --alpha=0 --beta=0.2 --init_way=Random


# reddit

python train_ctrl_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1  \ 
--lr_adj=0.1  --r=0.001 --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=0 --alpha=0 --beta=0.2 --init_way=K-means

python train_ctrl_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1  \ 
--lr_adj=0.1  --r=0.005 --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=0 --alpha=0 --beta=0.1 --init_way=K-means

python train_ctrl_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1  \ 
--lr_adj=0.1  --r=0.0005 --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=0 --alpha=0 --beta=0.1 --init_way=Random_real

