import os
import sys

in_fn = sys.argv[1]
out_fn = sys.argv[2]

fin = open(in_fn, 'r')
fout = open(out_fn, 'w')

for item in fin.readlines():
    data = item.rstrip().split()
    if len(data[-1]) == 0: data = data[:-1]
    if len(data) == 4:
        fout.write('%s %s\n' % (data[-1], data[-1]))
    else:
        fout.write('%s %s\n' % (data[-2], data[-1]))

fin.close()
fout.close()
