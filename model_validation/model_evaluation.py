import pandas as pd
import numpy as np
import os
import re

import constants


def post_process(data):
    actions = [i for i in range(1,11)]
    data = pd.read_csv(data)
    data.sort_values('seqNo',inplace=True)
    for i in actions:

        if i not in list(data['seqNo']):             
            emp_dic = {'start_time':[0.0],'end_time':[0.0],'action':['Action {}'.format(i)],
                       'score':[0],'seqNo':[i]}    
            data=pd.concat([data,pd.DataFrame(emp_dic)],axis=0)
            data.sort_values('seqNo',inplace=True)
            data.reset_index(inplace=True,drop=True)
    return data


def IOU_1D(f1, f2):
    ground, pred = pd.read_csv(f1), post_process(f2)

    print(">>>>>>>>>>. ground : ", ground)
    print(">>>>>>>>>>. pred : ", pred)

    a1 = [tuple(ground[['start_time', 'end_time']].iloc[i, :]) for i in range(len(ground))]
    a2 = [tuple(pred[['start_time', 'end_time']].iloc[i, :]) for i in range(len(pred))]

    print("a1 : ", a1)
    print("a2 : ", a2)

    print("a1.shape : ", len(a1))
    print("a2.shape : ", len(a2))

    assert len(a1) == len(a2)
    global s
    global iou
    l = []
    s = 0

    for i in range(len(a1)):
        tg1, tg2, t1, t2 = a1[i][0], a1[i][1], a2[i][0], a2[i][1]
        try:
            iou = max(0, min(tg2, t2) - max(tg1, t1)) / (max(tg2, t2) - (min(tg1, t1)))
        except:
            iou = 0
        s += iou
        l.append(iou)

    return s / len(a1)


def l1_norm(f1, f2):
    ground, pred = pd.read_csv(f1), post_process(f2)
    a1 = [tuple(ground[['start_time', 'end_time']].iloc[i, :]) for i in range(len(ground))]
    a2 = [tuple(pred[['start_time', 'end_time']].iloc[i, :]) for i in range(len(pred))]
    assert len(a1) == len(a2)
    global s
    global iou
    l = []
    s = 0
    for i in range(len(a1)):
        tg1, tg2, t1, t2 = a1[i][0], a1[i][1], a2[i][0], a2[i][1]
        l.append((abs(tg1 - t1) + abs(tg2 - t2)))

    return np.average(l)


def main():
    global I
    global L
    I, L = 0, 0
    g_files = [i for i in os.listdir(constants.valGTPath) if ('.csv' in i)]
    p_files = [i for i in os.listdir(constants.resultOutputPath) if ('.csv' in i)]
    g_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    p_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print("Files found : ", g_files, "    ", p_files)
    res = []
    result = {}

    for n, i in enumerate(g_files):
        res.append((IOU_1D(constants.valGTPath + '/' + i, constants.resultOutputPath + '/' + p_files[n]),
                    l1_norm(constants.valGTPath + '/' + i, constants.resultOutputPath + '/' + p_files[n])))

    for i in res:
        I += i[0]
        L += i[1]
    result['Accuracy'], result['Error'] = I / len(res), L / len(res)

    print(result)
    return result