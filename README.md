# NExT-OOD: Overcoming Dual Multiple-choice VQA Biases  (accepted to TPAMI 2023)

We construct a new videoQA benchmark **NExT-OOD** in OOD setting. 
The NExT-OOD can quantify models’ generalizability and measure their reasoning ability comprehensively.
It contains three sub-datasets including NExT-OOD-VA, NExT-OOD-QA, and NExT-OOD-VQA, which are designed for the vision-answer (VA) bias, question-answer (QA) bias, and VA&QA bias, respectively.

We also propose a graph-based cross-sample method for bias reduction,
which provides adequate debiasing guidance from the perspective of whole dataset, and encourages the model to focus on multimodal contents.

## Environment

Anaconda 4.9.2, python 3.6.8, pytorch 1.7.1 and cuda 11.0. For other libs, please refer to the file requirements.txt.

## Setup
```bash
# Create python environment (optional)
conda create -n nextood python=3.6.8
source activate nextood
git clone https://github.com/zhangxi1997/NExTOOD.git

# Install python dependencies
pip install -r requirements.txt
```

## Data Preparation
Please download the NExT-OOD datasets and pre-computed features from [here](https://drive.google.com/drive/folders/1VlQ8Pfpo0-a9pNcPuXh9yGpDEGjaDRaQ?usp=sharing). There are 4 zip files: 
- ```['NExT-OOD-VQA']```: NExT-OOD-VQA and NExT-OOD-VQA-auto dataset (for evaluation). 
- ```['NExT-OOD-VA']```: NExT-OOD-VA dataset (for evaluation).
- ```['NExT-OOD-QA']```: NExT-OOD-QA dataset (for evaluation).
- ```['NExT-QA']```: NExT-QA dataset (for training).
- ```['vid_feat.zip']```: Appearance and motion feature for video representation. (provided by [NExT-QA](https://github.com/doc-doc/NExT-QA)).
- ```['qas_bert_single.zip']```: Extracted BERT feature for QA-pair representation. (Based on [pytorch-pretrained-BERT](https://github.com/LuoweiZhou/pytorch-pretrained-BERT/)).
- ```['GCS.zip']```: Model checkpoint. 

After downloading the data, please put ```['NExT-OOD-VQA'],['NExT-OOD-VA'],['NExT-OOD-QA'],['NExT-QA']``` in the folder ```['dataset/NExT-OOD']```,
you will have directories like  ```['dataset/NExT-OOD/NExT-OOD-VQA']```.
Besides, please create a folder ```['dataset/NExT-OOD/feats']```, and unzip the video and QA features into it.
You will have directories like ```['dataset/NExT-OOD/feats/vid_feat/', 'dataset/NExT-OOD/feats/qas_bert/'``` in your workspace.
Please unzip the files in  ```['GCS.zip']``` into ```['models/']```. 

*(You are also encouraged to design your own pre-computed video features. In that case, please download the raw videos from [VidOR](https://xdshang.github.io/docs/vidor.html). 

## Usage
Once the data is ready, you can easily run the code.
First, to test the environment and code, we provide the prediction and model of the our proposed approach. 
You can get the results on the validation set of the NExT-OOD-VQA by running:
```
>./eval_NExTOOD_VQA.sh 0 #Evaluate the model with GPU id 0

(Expected output: "NExT-OOD-VQA best val: ('39', 42.71) , test result: ('test', 31.69)")

```
The command above will load the model under ['models/'] and generate the prediction file.
If you want to train the model, please run
```
>./main.sh 0 # Train the model with GPU id 0
```
It will train the model and save to ['models']. (*The results may be slightly different depending on the environments*)
After the training, to comprehensively evaluate the model on our NExT-OOD, please run
```
>./main_eval.sh 0 # Evaluate the model on our NExT-OOD with GPU id 0
```

## Citation
```
@article{zhang2023nextoood,
    author    = {Zhang, Xi and Zhang, Feifei and Xu, Changsheng},
    title     = {NExT-OOD: Overcoming Dual Multiple-choice VQA Biases},
    booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
    year      = {2023}
}
```

## Acknowledgement
Our model is based on the official [NExT-QA](https://github.com/doc-doc/NExT-QA) repository, we thank the authors to release their code. If you use the related part, please cite the corresponding paper commented in the code.
