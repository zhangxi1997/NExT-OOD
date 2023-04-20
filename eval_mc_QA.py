import os.path as osp
from utils import load_file
import json

map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}
map_vid_vidorID = 'dataset/NExT-OOD/NExT-QA/map_vid_vidorID.json'
with open(map_vid_vidorID) as f:
    map_vid_vidorID_list = json.load(f)

def accuracy_metric(sample_list_file, result_file):

    sample_list = load_file(sample_list_file)
    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    question = {}
    answers = {}

    for id, row in sample_list.iterrows():
        qns_id = str(row['video']) + '_' + str(row['qid'])
        qtype = str(row['type'])

        #(combine temporal qns of previous and next as 'TN')
        if qtype == 'TP': qtype = 'TN'
        group[qtype].append(qns_id)
        question[qns_id] = row['question']
        answers[qns_id] = str(row['a0']) + " %%% " + str(row['a1']) + " %%% " + str(row['a2']) + " %%% " + str(row['a3']) + " %%% " + str(row['a4'])

    preds = load_file(result_file)
    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    overall_acc = {'C':0, 'T':0, 'D':0}
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids: # video_id + ques_id

            cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']

            if answer == pred: 
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt


    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    for qtype in group_acc:
        print(map_name[qtype], end='\t')
    print('')
    print('')
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))
    return all_acc*100.0/all_cnt


def main(result_file, mode='val', type='OOD'):
    dataset_dir = 'dataset/NExT-OOD/'
    data_set = mode
    sample_list_file = osp.join(dataset_dir+'/NExT-OOD-QA', data_set + '_QA.csv')
    print('Evaluating {}'.format(result_file))

    acc = accuracy_metric(sample_list_file, result_file)
    return acc


if __name__ == "__main__":
    model_type = 'HGA'
    mode = 'val'
    model_prefix = 'bert-ft-h256-{}-example'.format(mode)
    result_file = 'results/{}-{}.json'.format(model_type, model_prefix)

    main(result_file, mode)
