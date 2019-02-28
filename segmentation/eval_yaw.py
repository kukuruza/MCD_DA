import argparse
import sqlite3
import logging
import numpy as np
from pprint import pformat

from tensorboard_logger import configure, log_value, log_histogram


def angle360_l1(inputs, targets):
#    print('Metrics: pred:', inputs, 'gt:', targets, 'diff:', out)
  diff = np.remainder(inputs - targets, 360.)
  return np.minimum(diff, 360. - diff)


parser = argparse.ArgumentParser('evaluateYaw')
parser.add_argument('--pred_db_file', required=True)
parser.add_argument('--gt_db_file', required=True)
parser.add_argument('--tflog_dir')
parser.add_argument('--epoch', type=int)
args = parser.parse_args()

conn = sqlite3.connect(args.pred_db_file)
c = conn.cursor()
c.execute('ATTACH ? AS "attached"', (args.gt_db_file,))
c.execute('SELECT pr.value, gt.value FROM properties pr INNER JOIN attached.properties gt '
          'WHERE pr.key == "yaw" AND gt.key == "yaw" AND pr.objectid == gt.objectid ORDER BY pr.objectid ASC')
entries = c.fetchall()
c.close()

logging.info ('Total %d objects in both the open and the ground truth databases.' % len(entries))
logging.debug (pformat(entries))

entries = [(-float(x), float(y)) for (x, y) in entries]  # "-" to fix a bug with sign in synthetic data.
pr_yaws, gt_yaws = zip(*entries)

metrics = angle360_l1(np.array(pr_yaws), np.array(gt_yaws))

bincounts, bin_edges = np.histogram(metrics, bins=24)
configure(args.tflog_dir, flush_secs=10)
step = args.epoch * 4168 // 10  # 8396
#print (step)
log_histogram('hist/test/yaw', (bin_edges, bincounts), step=step)
log_value('metrics/test/yaw', metrics.mean() * 10, step=step)
print (metrics.mean() * 10)

