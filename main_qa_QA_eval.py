from videoqa_GCS import *
from dataloader import sample_loader_QA as sample_loader
from build_vocab import Vocabulary
from utils import *
import argparse
import eval_mc_QA


NUM_THREADS = 1
torch.set_num_threads(NUM_THREADS)

def main(args, results):

    mode = args.mode
    if mode == 'train':
        batch_size = 64
        num_worker = 2
    else:
        batch_size = 64 #you may need to change to a number that is divisible by the size of test/val set, e.g., 4
        num_worker = 2

    model_type = 'GCS' #(GCS, EVQA, STVQA, CoMem, HME, HGA)
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
    use_bert = args.bert #True #Otherwise GloVe

    model_prefix = args.checkpoint

    checkpoint_path = 'models/' + args.checkpoint
    if os.path.exists(checkpoint_path) == False:
        os.mkdir(checkpoint_path)

    vis_step = 106 # visual step
    lr_rate = 5e-5 if use_bert else 1e-4
    epoch_num = 50

    data_loader = sample_loader.QALoader(batch_size, num_worker, video_feature_path, video_feature_cache,
                                      sample_list_path, vocab, use_bert, model_type, True, False)

    train_loader, val_loader, test_loader, val_loader_raw = data_loader.run(mode=mode)

    vqa = VideoQA(vocab, train_loader, val_loader, test_loader, glove_embed, use_bert, checkpoint_path, model_type,
                  model_prefix,
                  vis_step, lr_rate, batch_size, epoch_num, args.gin, args.delta, args.lambda1, args.lambda2)

    if args.epoch > 0:
        start = args.epoch
        end = args.epoch + 1
    else:
        start = 1  # model selection
        end = 50

    for epoch in range(start, end):
        for file in os.listdir(checkpoint_path):
            if file.split('-')[2] == str(epoch):
                model_file = file
                break

        if mode == 'val':
            result_file = f'results/QA_{model_type}-{model_prefix}-{mode}.json'
            # try:
            vqa.predict(model_file, result_file)
            balance_results = eval_mc_QA.main(result_file, mode)
            results[str(epoch)] = round(balance_results, 2)
            # except:
            #     print('Something wrong in epoch', epoch)

    sorted_results = sorted(results.items(), key=lambda kv: (kv[1], kv[0]))
    best = sorted_results[-1]

    # do test
    for file in os.listdir(checkpoint_path):
        if file.split('-')[3] == str(best[0]):
            model_file = file
            break
    mode = 'test'
    result_file = f'results/{model_type}-{model_prefix}-{mode}.json'
    vqa.predict_test(model_file, result_file)
    balance_results_test = eval_mc_QA.main(result_file, mode)

    sorted_results.append(('test',round(balance_results_test,2)))

    return sorted_results, model_type



if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.benchmark = True

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

    parser.add_argument('--gin', dest='gin', type=int,
                        default=3, help='Layer number of GIN')
    parser.add_argument('--delta', dest='delta', type=float,
                        default=0.5, help='parameter in GIN')
    parser.add_argument('--lambda1', dest='lambda1', type=float,
                        default=1.0, help='lambda1')
    parser.add_argument('--lambda2', dest='lambda2', type=float,
                        default=2.0, help='lambda2')

    parser.add_argument('--epoch', dest='epoch', type=int,
                        default=-1, help='epoch model for evaluation')
    args = parser.parse_args()

    start = time.time()
    results = {}
    results, model_type = main(args, results)

    print('------------------------NExT-OOD-QA----------------------------')
    print(results)
    print('best val:', results[-2], ', test result:', results[-1])