#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import os


in_csv = "/home/stat-caolei/code/SVS_SLIDES/CLAM/dataset_csv/fold1/test.csv"
out_csv = "./dataset_csv/test.csv"

df = pd.read_csv(in_csv)
df['slides_name'] = df['slides_name'].map(lambda x: x.split("/")[-1] + ".pt")
df['slides_name'] = df['slides_name'].map(lambda x: os.path.join("/home/stat-caolei/code/SVS_SLIDES/CLAM/tcga_dx_feat/", x))
df.to_csv(out_csv)
