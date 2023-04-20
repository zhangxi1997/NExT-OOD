from videoqa_GCS import *
from dataloader import sample_loader as sample_loader
from build_vocab import Vocabulary
from utils import *
import argparse
import eval_mc
import numpy as np


NUM_THREADS = 1
torch.set_num_threads(NUM_THREADS)


def main(args):

    mode = args.mode
    if mode == 'train':
        batch_size = 64
        num_worker = 2
    else:
        batch_size = 64 #you may need to change to a number that is divisible by the size of test/val set, e.g., 4
        num_worker = 2

    model_type = 'GCS'  #(GCS, EVQA, STVQA, CoMem, HME, HGA)
    if model_type == 'STVQA':
        spatial = True
    else:
        spatial = False # True for STVQA

    if spatial:
        #STVQA
        video_feature_path = 'dataset/feats/'
        video_feature_cache = 'dataset/feats/cache/'
    else:
        video_feature_cache = 'dataset/feats/cache/'
        video_feature_path = 'dataset/feats/'

    dataset = 'NExT-OOD'

    sample_list_path = 'dataset/{}/'.format(dataset)
    vocab = pkload('dataset/{}/vocab.pkl'.format(dataset))

    glove_embed = 'dataset/{}/glove_embed.npy'.format(dataset)
    use_bert = args.bert

    model_prefix = args.checkpoint

    checkpoint_path = '/data/zhangxi/videoqa_CS_saves/' + args.checkpoint
    if os.path.exists(checkpoint_path) == False:
        os.mkdir(checkpoint_path)

    vis_step = 106 # visual step
    lr_rate = 5e-5 if use_bert else 1e-4
    epoch_num = 50


    data_loader = sample_loader.QALoader(batch_size, num_worker, video_feature_path, video_feature_cache,
                                      sample_list_path, vocab, use_bert, model_type, True, False)

    train_loader, val_loader, test_loader = data_loader.run(mode=mode)

    vqa = VideoQA(vocab, train_loader, val_loader, test_loader, glove_embed, use_bert, checkpoint_path, model_type, model_prefix,
                  vis_step,lr_rate, batch_size, epoch_num, args.gin, args.delta, args.lambda1, args.lambda2)

    ep = 49
    acc = 43.67
    model_file = f'{model_type}-{model_prefix}-{ep}-{acc:.2f}.ckpt'


    if mode != 'train':
        mode = 'val'
        result_file = f'results/{model_type}-{model_prefix}-{mode}.json'
        vqa.predict(model_file, result_file)
        balance_results_raw = eval_mc.main(result_file, mode)
        print(balance_results_raw)

    else:
        #Model for resume-training.
        model_file = checkpoint_path + f'/{model_type}-{model_prefix}-43-39.67.ckpt'
        vqa.run(model_file, pre_trained=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', type=int,
                        default=0, help='gpu device id')
    parser.add_argument('--mode', dest='mode', type=str,
                        default='train', help='train or val')
    parser.add_argument('--bert', dest='bert', action='store_true',
                        help='use bert or glove')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str,
                        default='ck', help='checkpoint name')
    parser.add_argument('--object', dest='object', type=int,
                        default=0, help='object idx (0-80)')
    parser.add_argument('--seed', type=int,
                        default=666, help='random seed')

    parser.add_argument('--gin', dest='gin', type=int,
                        default=3, help='Layer number of GIN')
    parser.add_argument('--delta', dest='delta', type=float,
                        default=0.5, help='parameter in GIN')
    parser.add_argument('--lambda1', dest='lambda1', type=float,
                        default=1.0, help='lambda1')
    parser.add_argument('--lambda2', dest='lambda2', type=float,
                        default=2.0, help='lambda2')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    main(args)


