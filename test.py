import csv
import os

with open(os.path.join('results/Fraud/greedy/1_less_feature', 'metrics.csv'), 'a') as f:
    w = csv.writer(f)
    w.writerow([1, 2, 3, 4, 5])