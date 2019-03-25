import os
import sys
import json

if len(sys.argv) != 3:
    print('Example Usage: python merge_result_json.py Knife 42')
    exit(1)

in_cat = sys.argv[1]
in_anno = sys.argv[2]

in_dir = '../data/%s' % in_anno
in_res_json_fn = os.path.join(in_dir, 'result.json')
with open(in_res_json_fn, 'r') as fin:
    data = json.load(fin)

trans_fn = '../stats/merging_hierarchy_mapping/%s.txt' % in_cat
with open(trans_fn, 'r') as fin:
    node_mapping = {item.rstrip().split()[0]: item.rstrip().split()[1] for item in fin.readlines()}

new_template_fn = '../stats/after_merging_label_ids/%s.txt' % in_cat
new_part2type = dict();
with open(new_template_fn, 'r') as fin:
    for item in fin.readlines():
        x, y, z = item.rstrip().split()
        new_part2type[y] = z;

new_result = []; cur_id = 0;

def get_all_objs(record):
    if 'children' in record.keys():
        out = []
        for item in record['children']:
            out += get_all_objs(item)
        return out
    elif 'objs' in record.keys():
        return record['objs']
    else:
        print('ERROR: no children key nor objs key! %s %s' % (in_cat, in_anno))
        exit(1)

def traverse(record, cur_name, cur_node):
    global new_result, cur_id

    if len(cur_name) == 0:
        cur_name = record['name']
    else:
        cur_name = cur_name + '/' + record['name']

    if cur_name in node_mapping.keys():
        new_part = node_mapping[cur_name]
        history_name = '/'.join(new_part.split('/')[:-1])
        cur_new_name = new_part.split('/')[-1]

        new_node = {'name': cur_new_name, 'text': cur_new_name, 'id': cur_id, 'ori_id': record['id']}
        cur_id += 1

        if cur_node is None:
            new_result.append(new_node)
        elif 'children' in cur_node.keys():
            cur_node['children'].append(new_node)
        else:
            cur_node['children'] = [new_node]
        
        new_node['objs'] = get_all_objs(record)

        if 'children' in record.keys():
            for item in record['children']:
                traverse(item, cur_name, new_node)
    else:
        if 'children' in record.keys():
            for item in record['children']:
                traverse(item, cur_name, cur_node)

traverse(data[0], '', None)

out_fn = os.path.join(in_dir, 'result_after_merging.json')
with open(out_fn, 'w') as fout:
    json.dump(new_result, fout)

