import os.path as osp
from utils import load_file
import json
import numpy as np

map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}
map_vid_vidorID = 'dataset/NExT-OOD/NExT-QA/map_vid_vidorID.json'
with open(map_vid_vidorID) as f:
    map_vid_vidorID_list = json.load(f)

def accuracy_metric(sample_list_file, result_file):

    sample_list = load_file(sample_list_file)
    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    question = {}
    answers = {}

    obj_group = {}
    for i in range(1, 81):
        obj_group[str(i)] = []

    for id, row in sample_list.iterrows():
        qns_id = str(row['video']) + '_' + str(row['qid'])
        qtype = str(row['type'])

        #(combine temporal qns of previous and next as 'TN')
        if qtype == 'TP': qtype = 'TN'
        group[qtype].append(qns_id)
        question[qns_id] = row['question']
        answers[qns_id] = str(row['a0']) + " %%% " + str(row['a1']) + " %%% " + str(row['a2']) + " %%% " + str(row['a3']) + " %%% " + str(row['a4'])

        category = row['cate'].split()
        for obj in category:
            obj_group[str(obj)].append(qns_id)

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
        for qid in qns_ids:

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
    # for qtype, acc in group_acc.items():
    #     print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
    print('')
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))

    obj_group_Acc = {}
    for i in range(1, 81):
        obj_group_Acc[str(i)] = 0

    for obj_type, qns_ids in obj_group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids: # video_id + ques_id
            cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']

            if answer == pred:
                acc += 1
        if cnt != 0:
            obj_group_Acc[obj_type] = round(100*acc / cnt, 4)
        else:
            obj_group_Acc[obj_type] = -1

        all_cnt += cnt

    # print('object_group_Acc:',obj_group_Acc)
    obj_acc_list = []
    for _, value in obj_group_Acc.items():
        if value!=-1:
            obj_acc_list.append(value)

    # print(all_cnt)
    print('Avg object acc: {:.4f}'.format(np.mean(obj_acc_list)))
    # avg_acc = round(np.mean(obj_acc_list), 4)
    # obj_group_Acc['avg'] = avg_acc


def main(result_file, mode='val'):
    dataset_dir = 'dataset/NExT-OOD/NExT-QA'
    data_set = mode
    sample_list_file = osp.join(dataset_dir, data_set + '_parse.csv')
    print('Evaluating {}'.format(result_file))
    accuracy_metric(sample_list_file, result_file)


if __name__ == "__main__":
    model_type = 'HGA'
    mode = 'val'
    model_prefix = 'bert-ft-h256-{}-example'.format(mode)
    result_file = 'results/{}-{}.json'.format(model_type, model_prefix)

    main(result_file, mode)
