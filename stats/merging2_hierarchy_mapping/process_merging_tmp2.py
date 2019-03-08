import os
import sys

in_fn = sys.argv[1]
out_fn = sys.argv[2]

fin = open(in_fn, 'r')
fout = open(out_fn, 'w')

res = dict()
def add(d):
    global res
    l = d.split('/')
    cur_res = res
    for i in l:
        if i not in cur_res.keys():
            cur_res[i] = dict()
        cur_res = cur_res[i]

data_list = []
for item in fin.readlines():
    data = item.rstrip().split()[-1]
    data_list.append(data)
    add(data)

cur_id = 0
def traverse(n, d, cur_name):
    global cur_id
    cur_id += 1

    if cur_name == None:
        cur_cur_name = n
    else:
        cur_cur_name = cur_name + '/' + n

    assert cur_cur_name in data_list, cur_cur_name

    if len(d.keys()) == 0:
        fout.write('%d %s leaf\n' % (cur_id, cur_cur_name))
    else:
        fout.write('%d %s subcomponents\n' % (cur_id, cur_cur_name))
        for item in d.keys():
            traverse(item, d[item], cur_cur_name)

assert len(res.keys()) == 1
root = res.keys()[0]
traverse(root, res[root], None)
