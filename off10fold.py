import os
import sys


name = sys.argv[1]
dataset = sys.argv[2]
batch_size = sys.argv[3]
compute_size = sys.argv[4]
step = sys.argv[5]
gd = sys.argv[6]
cccp = sys.argv[7]

for i in range(0, 3):
 os.system("python offrun.py " + name + str(i) + " " + dataset  + " " + batch_size + " " + compute_size + " " + step + " " + gd + " " + cccp)


