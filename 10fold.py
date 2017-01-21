import os
import sys


name = sys.argv[1]
dataset = sys.argv[2]
batch_size = sys.argv[3]
step = sys.argv[4]
gd = sys.argv[5]

for i in range(5, 10):
 os.system("python run.py " + name + str(i) + " " + dataset + " " + batch_size + " " + step + " " + gd)

