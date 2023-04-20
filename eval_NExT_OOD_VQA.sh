#########################################################################
# File Name: eval_NExTOOD_vqa.sh
# Author: Xi Zhang
# mail: zhangxi2019@ia.ac.cn
#########################################################################
#!/bin/bash

source activate videoqa

GPU=$1
export CUDA_VISIBLE_DEVICES=$GPU

# the para "N" means the construction parameter in the paper,
# '--N N1' means evaluate the model on the dataset with N=1
python main_qa_VAQA_eval.py --mode val --gpu $GPU --checkpoint GCS --epoch 39 --N N1

#python main_qa_VAQA_balance_ours2.py --mode val --gpu $GPU --checkpoint GCS --K K1_auto


