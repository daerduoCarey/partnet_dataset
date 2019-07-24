import os
import sys
import json
from subprocess import call

in_dir = '../Chair'
tot = 0
for item in os.listdir(in_dir):
    if not item.startswith('.'):
        cur_dir = os.path.join(in_dir, item)
        x = [item for item in os.listdir(os.path.join(cur_dir, 'leaf_part_obj')) if not item.startswith('.')]
        if len(x) == 1:
            tot += 1
            cmd = 'bash gen_html_view.sh %s' % (cur_dir)
            print tot, item, cmd
            call(cmd, shell=True)



