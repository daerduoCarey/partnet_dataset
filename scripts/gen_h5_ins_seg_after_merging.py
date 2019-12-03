"""
This file tries to load the PartNet data and export hdf5 files for network training and testing.
For each data record, it loads the data from result.json and objs folder and reads the merging criterion
to load in after-merging PartNet part semantics that are used for Sec 5.1, 5.2 and 5.3.

"""

import os
import sys
from geometry_utils import *
import numpy as np
import json
from progressbar import ProgressBar
import copy

in_cat = sys.argv[1]
split = sys.argv[2]
print(in_cat, split)

in_fn = '../stats/train_val_test_split/%s.%s.json' % (in_cat, split)
with open(in_fn, 'r') as fin:
    item_list = json.load(fin)
in_fn = '../stats/merging_hierarchy_mapping/%s.txt' % in_cat
with open(in_fn, 'r') as fin:
    node_mapping = {d.rstrip().split()[0]: d.rstrip().split()[1] for d in fin.readlines()}

def load_file(fn):
    with open(fn, 'r') as fin:
        lines = [line.rstrip().split() for line in fin.readlines()]
        pts = np.array([[float(line[0]), float(line[1]), float(line[2])] for line in lines], dtype=np.float32)
        nor = np.array([[float(line[3]), float(line[4]), float(line[5])] for line in lines], dtype=np.float32)
        rgb = np.array([[int(line[6]), int(line[7]), int(line[8])] for line in lines], dtype=np.float32)
        opacity = np.array([float(line[9]) for line in lines], dtype=np.float32)
        return pts, nor, rgb, opacity

def load_label(fn):
    with open(fn, 'r') as fin:
        label = np.array([int(item.rstrip()) for item in fin.readlines()], dtype=np.int32)
        return label

def save_h5(fn, pts, nor, rgb, opacity, label):
    fout = h5py.File(fn, 'w')
    fout.create_dataset('pts', data=pts, compression='gzip', compression_opts=4, dtype='float32')
    fout.create_dataset('nor', data=nor, compression='gzip', compression_opts=4, dtype='float32')
    fout.create_dataset('rgb', data=rgb, compression='gzip', compression_opts=4, dtype='uint8')
    fout.create_dataset('opacity', data=opacity, compression='gzip', compression_opts=4, dtype='float32')
    fout.create_dataset('label', data=label, compression='gzip', compression_opts=4, dtype='int32')
    fout.close()

def save_json(fn, data):
    with open(fn, 'w') as fout:
        json.dump(data, fout)

def get_all_leaf_ids(record):
    if 'children' in record.keys():
        out = []
        for item in record['children']:
            out += get_all_leaf_ids(item)
        return out
    elif 'objs' in record.keys():
        return [record['id']]
    else:
        print('ERROR: no children key nor objs key! %s %s' % (in_cat, in_anno))
        exit(1)

def traverse(record, cur_name):
    global new_result
    if len(cur_name) == 0:
        cur_name = record['name']
    else:
        cur_name = cur_name + '/' + record['name']
    if cur_name in node_mapping.keys():
        new_part_name = node_mapping[cur_name]
        leaf_id_list = get_all_leaf_ids(record)
        new_result.append({'leaf_id_list': leaf_id_list, 'part_name': new_part_name})
    if 'children' in record.keys():
        for item in record['children']:
            traverse(item, cur_name)

def normalize_pc(pts):
    x_max = np.max(pts[:, 0]); x_min = np.min(pts[:, 0]); x_mean = (x_max + x_min) / 2;
    y_max = np.max(pts[:, 1]); y_min = np.min(pts[:, 1]); y_mean = (y_max + y_min) / 2;
    z_max = np.max(pts[:, 2]); z_min = np.min(pts[:, 2]); z_mean = (z_max + z_min) / 2;
    pts[:, 0] -= x_mean;
    pts[:, 1] -= y_mean;
    pts[:, 2] -= z_mean;
    scale = np.sqrt(np.max(np.sum(pts**2, axis=1)))
    pts /= scale
    return pts


n_shape = 1024
n_point = 10000

batch_pts = np.zeros((n_shape, n_point, 3), dtype=np.float32)
batch_nor = np.zeros((n_shape, n_point, 3), dtype=np.float32)
batch_rgb = np.zeros((n_shape, n_point, 3), dtype=np.uint8)
batch_opacity = np.zeros((n_shape, n_point), dtype=np.float32)
batch_label = np.zeros((n_shape, n_point), dtype=np.int32)
batch_record = []

bar = ProgressBar()
k = 0; t = 0;
for item_id in bar(range(len(item_list))):
    item = item_list[item_id]

    in_fn = '../data/%s/point_sample/sample-points-all-pts-nor-rgba-10000.txt' % item['anno_id']
    assert os.path.exists(in_fn)
    in_res_fn = '../data/%s/result.json' % item['anno_id']
    assert os.path.exists(in_res_fn)
    in_label_fn = '../data/%s/point_sample/sample-points-all-label-10000.txt' % item['anno_id']
    assert os.path.exists(in_label_fn)

    pts, nor, rgb, opacity = load_file(in_fn)
    pts = normalize_pc(pts)
    old_label = load_label(in_label_fn)

    with open(in_res_fn, 'r') as fin:
        data = json.load(fin)
    new_result = [];
    traverse(data[0], '')

    new_record = copy.deepcopy(item)
    new_record['ins_seg'] = new_result

    batch_pts[k, ...] = pts
    batch_nor[k, ...] = nor
    batch_rgb[k, ...] = rgb
    batch_opacity[k, ...] = opacity
    batch_label[k, :] = old_label
    batch_record.append(new_record)
    k += 1

    if k == n_shape or item_id+1 == len(item_list):
        out_dir = 'ins_seg_h5/%s' % in_cat
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_fn_prefix = os.path.join(out_dir, '%s-%02d' % (split, t))
        save_h5(out_fn_prefix+'.h5', batch_pts[:k, ...], batch_nor[:k, ...], batch_rgb[:k, ...], \
                batch_opacity[:k, ...], batch_label[:k, ...])
        save_json(out_fn_prefix+'.json', batch_record)
        t += 1; k = 0; batch_record = [];


