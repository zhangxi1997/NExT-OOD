#########################################################################
# File Name: GCS.sh
# Author: Xi Zhang
# mail: zhangxi2019@ia.ac.cn
#########################################################################
#!/bin/bash

source activate videoqa

GPU=$1
export CUDA_VISIBLE_DEVICES=$GPU
ck=GCS_saves

#for seed in {664,665,666,667,668}
#do
#  echo "seed:"$seed
#  ck='GCS_'$seed
#  echo $ck
#
#  mkdir LOG/$ck

python main_qa.py --mode train --gpu $GPU --checkpoint $ck

#done
