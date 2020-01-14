#coding=utf-8
#-*- coding: UTF-8 -*-
import sys

"""python3"""


import re
import os
import json
import time
import codecs
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from tqdm import trange
from collections import Counter


codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)


data = json_load("personachat_self_original.json")['train'][:30]
print("loaded")
json_dump(data, "personachat.json")