#########################################################################
# File Name: GCS.sh
# Author: Xi Zhang
# mail: zhangxi2019@ia.ac.cn
#########################################################################
#!/bin/bash

source activate videoqa

GPU=$1
export CUDA_VISIBLE_DEVICES=$GPU
ck=GCS

#for seed in {664,665,666,667,668}
#do
#  echo "seed:"$seed
#  ck='GCS_'$seed
#  echo $ck
#
#  mkdir LOG/$ck

# the para "N" means the construction parameter in the paper,
# '--N all' means evaluate the model on the dataset with N=1, N=2, and N=5,
# '--N N1' means evaluate the model on the dataset with N=1
python main_qa_VAQA_eval.py --mode val --gpu $GPU --checkpoint $ck --N all --epoch 39

python main_qa_VAQA_eval.py --mode val --gpu $GPU --checkpoint $ck --N all_auto --epoch 39

python main_qa_VA_eval.py --mode val --gpu $GPU --checkpoint $ck --N all --epoch 39

python main_qa_QA_eval.py --mode val --gpu $GPU --checkpoint $ck --epoch 39

#done
