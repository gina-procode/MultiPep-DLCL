#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/16 14:35
# @Author : lt,fhh
# @FileName: __init__.py.py
# @Software: PyCharm
import csv
import os
import time
import torch
import pandas as pd
import estimate
from config import pep_config
from models.DLCL import DLCL
from DataLoad import data_load
from train import DataTrain, predict, CosineScheduler
import numpy as np
import  random
import sys
def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录到Python路径
sys.path.append(current_dir)
get_random_seed(20230226)
torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RMs = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP',
       'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']
title1 = ['Model', "Loss", 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime',
          'Test_Time']


def spent_time(start, end):
    epoch_time = end - start
    minute = int(epoch_time / 60)
    secs = int(epoch_time - minute * 60)
    return minute, secs


def save_results(model_name, loss_name, start, end, test_score, title, file_path):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if len(test_score) != 21:
        content = [[model_name, loss_name,
                    '%.3f' % test_score[0],
                    '%.3f' % test_score[1],
                    '%.3f' % test_score[2],
                    '%.3f' % test_score[3],
                    '%.3f' % test_score[4],
                    '%.3f' % (end - start),
                    now]]
    else:
        title.append('Model')
        content1 = [f'{i:.3f}' for i in test_score]
        content1.append(model_name)
        content = [content1]

    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None, encoding='gbk')
        one_line = list(data.iloc[0])
        if one_line == title:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerows(content)
        else:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)
                writer.writerows(content)
    else:
        with open(file_path, 'a+', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)
def main():
    args = pep_config.get_config()
    print("The current task is: Multifunctional therapeutic peptides recognition")
    result_dir = os.path.join(current_dir, 'result')
    saved_models_dir = os.path.join(current_dir, 'saved_models')
    os.makedirs(result_dir, exist_ok=True)  # 自动创建result目录
    os.makedirs(saved_models_dir, exist_ok=True)  # 自动创建saved_models目录
    models_file = os.path.join(result_dir, f'{args.task}_models.txt')
    Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parse_file = os.path.join(result_dir, f'{args.task}_pares.txt')
    file1 = open(parse_file, 'a')
    file1.write(Time)
    file1.write('\n')
    print(args, file=file1)
    file1.write('\n')
    file1.close()
    file_path = os.path.join(result_dir, 'model_select.csv')
    print("\n Data is loading...")
    # ====== 新增数据集路径配置 ======
    dataset_dir = os.path.join(current_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)  # 确保dataset目录存在
    
    # ====== 修改数据路径指向 ======
    args.train_direction = os.path.join(dataset_dir, 'train.txt')  # 自动生成正确路径
    args.test_direction = os.path.join(dataset_dir, 'test.txt')
    train_datasets, test_datasets, subtests, _ = data_load(
        batch=args.batch_size,
        train_direction=args.train_direction,
        test_direction=args.test_direction,
        subtest=args.subtest,
        CV=False
    )
    print("\n Data is loaded...")
    get_random_seed(20230226)
    torch.backends.cudnn.deterministic = True
    print("\n" + "="*50)
    print("Starting to train a new model…")
    test_score, aim, cov, acc, ab_true, ab_false = [], 0, 0, 0, 0, 0
    start_time = time.time()

    for i in range(len(train_datasets)):
        train_dataset = train_datasets[i]
        test_dataset = test_datasets[i]
        train_start = time.time()
        model=DLCL()
        model_name = model.__class__.__name__
        title_task = f"{args.task}+{model_name}"


        model_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        file2 = open(models_file, 'a')
        file2.write(model_time)
        file2.write('\n')
        print(model, file=file2)
        file2.write('\n')
        file2.close()
        loss_name = None
        # args.model_path = f"./saved_models/th_cv+TC_CBAM{i}.pth"

        print(f"{model_name} is training......")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        lr_scheduler = CosineScheduler(250, base_lr=args.learning_rate, warmup_steps=20)
        loss_name = ""
        for criterion in args.criterion:
            loss_name += criterion.__class__.__name__
        Train = DataTrain(model, optimizer, args.criterion, lr_scheduler, device=DEVICE)

        Train.train_step(train_dataset, args.epochs, model_name, args.alpha, args.beta)

        each_model = os.path.join(saved_models_dir, f'{title_task}{i}.pth')
        torch.save(model.state_dict(), each_model)

        model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)

        test_score = estimate.evaluate(model_predictions, true_labels, threshold=args.threshold)

        test_end = time.time()
        save_results(title_task, loss_name, train_start, test_end, test_score, title1, file_path)

        run_time = time.time()
        m, s = spent_time(start_time, run_time)
        print(f"{args.task}, {model_name}'s runtime:{m}m{s}s")
        print("Result of the newly trained model is:：")
        print(f'aiming: {test_score[0]:.3f}')
        print(f'coverage: {test_score[1]:.3f}')
        print(f'accuracy: {test_score[2]:.3f}')
        print(f'absolute_true: {test_score[3]:.3f}')
        print(f'absolute_false: {test_score[4]:.3f}\n')
        # MCC = test_score[5]
        # for h in range(len(MCC)):
        #     print(f"{RMs[h]}'s MCC:{MCC[h]}")
        if subtests:
            for subtest in subtests:
                sub_predictions, sub_true_labels = predict(model, subtest, device=DEVICE)
                subtest_score = estimate.evaluate(sub_predictions, sub_true_labels, args.threshold)
                aim += subtest_score[0]
                cov += subtest_score[1]
                acc += subtest_score[2]
                ab_true += subtest_score[3]
                ab_false += subtest_score[4]
                subtest_end = time.time()
                save_results(title_task, loss_name, train_start, subtest_end, subtest_score, title1, file_path)
            print("test subset：")
            print(f'aiming: {aim / len(subtests):.3f}')
            print(f'coverage: {cov / len(subtests):.3f}')
            print(f'accuracy: {acc / len(subtests):.3f}')
            print(f'absolute_true: {ab_true / len(subtests):.3f}')
            print(f'absolute_false: {ab_false / len(subtests):.3f}\n')
            test_score = [aim / len(subtests), cov / len(subtests), acc / len(subtests), ab_true / len(subtests),
                          ab_false / len(subtests)]
            save_results('average', None, start_time, run_time, test_score, title1, file_path)
if __name__ == '__main__':
    main()