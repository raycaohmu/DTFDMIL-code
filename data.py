#!/usr/bin/env python
# coding=utf-8
import torch
import pandas as pd

import sys


def reOrganize_mDATA(csv_path):
    FeatList = []
    SlideNames = []
    Labels = []
    df = pd.read_csv(csv_path, index_col=[0])
    for i, (x, y) in df.iterrows():
        sys.stdout.write('\r' + " load:%.2f%%" % float(i / len(df) * 100))
        feat = torch.load(x)[:, :-2].float()
        FeatList.append(feat)
        SlideNames.append(x)
        Labels.append(y)

    return SlideNames, FeatList, Labels


if __name__ == "__main__":
    reOrganize_mDATA("./train_xiugao.csv")
