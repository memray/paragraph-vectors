# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import string

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


import operator
import os
import argparse
import json
import sys
import csv
import re

input_file_path ='/Users/memray/Data/academic/MicrosoftAcademicGraph/mag_fos=ir_en.txt'
output_file_path='../../data/doc2vec-pytorch_mag_fos=ir.csv'

print(os.getcwd())


count = 0
no_key_count = 0
no_abs_count = 0
output_count = 0

keyword_dict = {}

output_file     = open(output_file_path, 'w')
csv_writer      = csv.writer(output_file, quoting=csv.QUOTE_ALL)

with open(input_file_path, 'r') as input_file:
    for line in input_file:
        count += 1
        if count % 10000 == 0:
            print('Processing %s - %d' % (input_file_path, count))

        j = json.loads(line)
        if 'keywords' not in j or len(j['keywords']) == 0 or 'abstract' not in j or j['abstract'].strip() == '':
            if 'keywords' not in j or len(j['keywords']) == 0:
                no_key_count += 1
            if 'abstract' not in j or j['abstract'].strip() == '':
                no_abs_count += 1
        else:
            keywords = [k.lower().replace(' ', '_') for k in j['keywords']]
            [operator.setitem(keyword_dict, k, keyword_dict.get(k, 0) + 1) for k in keywords]

            words    = j['title'].lower().split() + re.sub('(#+\w+)+', ' ', j['abstract'].lower()).split()

            # remove all the non-ascii characters
            text     = ''.join(x for x in ' '.join(words) if x in string.printable)
            # text     = bytes(' '.join(words), 'utf-8').decode('utf-8', 'ignore')

            # if text.find('#') > 0:
            #     print(text)
            csv_writer.writerow([text])
            output_count += 1

output_file.close()

print('[Info] Dumping the processed data to new text file', output_file_path)
print('[Info] %d/%d valid docs' % (output_count, count))
print('[Info] %d documents with no keywords, %d with no abstract' % (no_key_count, no_abs_count))