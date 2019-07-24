import argparse
import os
import sys
import json
import scipy.misc as misc
import numpy as np

in_dir = sys.argv[1]

# write the file header and footer
html_head = '<html><head><meta charset="UTF-8"><title>Simple Viewer</title>' + \
        '<style>table {table-layout: fixed; }th, td { width: 100px; }</style></head><body>' 

html_tail = '</body></html>'

def gen_html_for_tree_hier(html_fn, tree_hier, parts_render_dir):
    fout = open(html_fn, 'w')
    fout.write(html_head+'\n')

    node_level = {}; node_loc = {}; all_nodes = [];
    def find_level_loc(cur_tree_hier, cur_level, cur_loc):
        node_id = cur_tree_hier['id']
        all_nodes.append(node_id)

        if 'children' in cur_tree_hier:
            child_nodes = cur_tree_hier['children']
        else:
            child_nodes = []

        if cur_level not in node_level.keys():
            node_level[cur_level] = []

        node_level[cur_level].append(node_id)
        if len(child_nodes) == 0:
            return 1
        else:
            old_cur_loc = cur_loc
            for child_node in child_nodes:
                child_loc = find_level_loc(child_node, cur_level+1, cur_loc)
                node_loc[child_node['id']] = cur_loc
                cur_loc += child_loc + 1
            return cur_loc - old_cur_loc

    root_node = tree_hier['id']

    node_loc[root_node] = 0
    find_level_loc(tree_hier, 0, 0)

    max_level = max(node_level.keys())

    fout.write('<table>')
    tot_parts = 0
    for level_id in range(max_level+1):
        fout.write('<tr>')
        cur_level_node_locs = {node_loc[item]: item for item in node_level[level_id]}
        cur_level_locs_dict = cur_level_node_locs.keys()
        tot_parts += len(cur_level_locs_dict)
        max_loc = max(cur_level_locs_dict)
        for width_id in range(max_loc+1):
            if width_id in cur_level_locs_dict:
                cur_part_img = os.path.join('parts_render/', str(cur_level_node_locs[width_id])+'.png')
                cur_meta_file = os.path.join(in_dir, 'parts_render/', str(cur_level_node_locs[width_id])+'.txt')
                with open(cur_meta_file, 'r') as fin:
                    meta = fin.readlines()[0].rstrip();
                fout.write('<td><p>%s</p><a href="%s"><img src="%s" width="100px" height="100px"/></a></td>'%(meta, cur_part_img, cur_part_img))
            else:
                fout.write('<td></td>')
        fout.write('</tr>')
    fout.write('</table>')

    fout.write(html_tail)
    fout.close()


model_path = in_dir
tree_hier_json = os.path.join(model_path, 'result.json')
parts_render_dir = os.path.join(model_path, 'parts_render')

with open(tree_hier_json, 'r') as fin:
    tree_hier = json.load(fin)[0]

html_fn = os.path.join(model_path, 'tree_hier.html')
gen_html_for_tree_hier(html_fn, tree_hier, parts_render_dir)

