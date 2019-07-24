# -*- coding: utf-8 -*-

import argparse
import random
import time
import scipy.misc as misc
import json
import os
import sys
import numpy as np
from subprocess import call
from collections import deque

in_dir = sys.argv[1]

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)

    return v, f

def export_obj(out, v, f, color):
    mtl_out = out.replace('.obj', '.mtl')

    with open(out, 'w') as fout:
        fout.write('mtllib %s\n' % mtl_out)
        fout.write('usemtl m1\n')
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    with open(mtl_out, 'w') as fout:
        fout.write('newmtl m1\n')
        fout.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout.write('Ka 0 0 0\n')

    return mtl_out

def render_mesh(v, f, color=[0.216, 0.494, 0.722]):
    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    tmp_obj = os.path.join(tmp_dir, str(time.time()).replace('.', '_')+'_'+str(random.random()).replace('.', '_')+'.obj')
    tmp_png = tmp_obj.replace('.obj', '.png')

    tmp_mtl = export_obj(tmp_obj, v, f, color=color)

    cmd = 'bash renderer/render.sh renderer/model.blend %s %s' % (tmp_obj, tmp_png)
    call(cmd, shell=True)

    img = misc.imread(tmp_png)
    img = img[1:401, 350: 750, :]
    img = img.astype(np.float32)

    all_white = np.ones((img.shape), dtype=np.float32) * 255

    img_alpha = img[:, :, 3] * 1.0 / 256
    all_white_alpha = 1.0 - img_alpha

    all_white[:, :, 0] *= all_white_alpha
    all_white[:, :, 1] *= all_white_alpha
    all_white[:, :, 2] *= all_white_alpha

    img[:, :, 0] *= img_alpha
    img[:, :, 1] *= img_alpha
    img[:, :, 2] *= img_alpha

    out = img[:, :, :3] + all_white[:, :, :3]

    cmd = 'rm -rf %s %s %s' % (tmp_obj, tmp_png, tmp_mtl)
    call(cmd, shell=True)

    return out


cur_shape_dir = in_dir
cur_part_dir = os.path.join(cur_shape_dir, 'objs')
leaf_part_ids = [item.split('.')[0] for item in os.listdir(cur_part_dir) if item.endswith('.obj')]

cur_render_dir = os.path.join(cur_shape_dir, 'parts_render')
if not os.path.exists(cur_render_dir):
    os.mkdir(cur_render_dir)

root_v_list = []; root_f_list = []; tot_v_num = 0;
for idx in leaf_part_ids:
    v, f = load_obj(os.path.join(cur_part_dir, str(idx)+'.obj'))
    mesh = dict();
    mesh['v'] = v; mesh['f'] = f;
    root_v_list.append(v);
    root_f_list.append(f+tot_v_num);
    tot_v_num += v.shape[0];

root_v = np.vstack(root_v_list)
root_f = np.vstack(root_f_list)

center = np.mean(root_v, axis=0)
root_v -= center
scale = np.sqrt(np.max(np.sum(root_v**2, axis=1))) * 1.5
root_v /= scale

root_render = render_mesh(root_v, root_f)

cur_result_json = os.path.join(cur_shape_dir, 'result.json')
with open(cur_result_json, 'r') as fin:
    tree_hier = json.load(fin)[0]

def render(data):
    cur_v_list = []; cur_f_list = []; cur_v_num = 0;
    if 'objs' in data.keys():
        for child in data['objs']:
            v, f = load_obj(os.path.join(cur_part_dir, child+'.obj'))
            v -= center
            v /= scale
            cur_v_list.append(v)
            cur_f_list.append(f+cur_v_num)
            cur_v_num += v.shape[0]
    elif 'children' in data.keys():
        for child in data['children']:
            v, f = render(child)
            cur_v_list.append(v)
            cur_f_list.append(f+cur_v_num)
            cur_v_num += v.shape[0]
    else:
        return

    part_v = np.vstack(cur_v_list)
    part_f = np.vstack(cur_f_list)

    part_render = render_mesh(part_v, part_f, color=[0.93, 0.0, 0.0])
    alpha_part = 0.3 * root_render + 0.7 * part_render
    out_filename = os.path.join(cur_render_dir, str(data['id'])+'.png')
    misc.imsave(out_filename, alpha_part)
    out_meta_fn = os.path.join(cur_render_dir, str(data['id'])+'.txt')
    with open(out_meta_fn, 'w') as fout:
        fout.write(u' '.join((str(data['id']), data['name'], data['text'])).encode('utf-8').strip())

    return part_v, part_f

render(tree_hier)
