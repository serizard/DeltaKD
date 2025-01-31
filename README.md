# AAAKD (1st brAIn research program)
Attention-Aware Adaptive Knowledge Distillation for Vision Transformers

## Contributors:
- Semin Kim
- Jeonghwan Cho
- Seonah Ryu
- Xu Meow


## Install
1. Clone this repo and run the commands below
```
https://github.com/serizard/AAAKD.git && cd AAAKD 
```

2. Install Packages
```
conda create -n aaakd python=3.10 -y
conda activate aaakd
pip install -e .
```


## Run experiments
```
cd AAAKD
bash exp/baseline-deit-tiny.sh $GPU_IDS
```

GPU_IDS -> ex: 0,1,2,3 (GPU ids to use)

Feel free to customize other arguments and configs.
