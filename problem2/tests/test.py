# Course: Computer Vision and Artificial Intelligence for Autonomous Cars, ETH Zurich
# Material for Project 2
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import os, sys, argparse
import pathlib
import psutil
import yaml
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timeit import default_timer as timer
import numpy as np
import torch

from dataset import DatasetLoader

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_path', default='config.yaml')
parser.add_argument('--recordings_dir', default='tests/recordings')
parser.add_argument('--task', type=int)
args = parser.parse_args()

ERROR_TRESHOLD = 0.002
DURATION_THRESHOLD = 1.3
LOSS_THRESHOLD = 1e-7
task2_idx = 266

class CheckTest():
	def __init__(self, config, recordings_dir):
		self.config, self.recordings_dir = config, recordings_dir
		self.ds = DatasetLoader(config['data'], 'val')

	def display_test_result(self, result, duration=False):
		result_message = 'Test passed.' if result else 'Test failed.'
		print(result_message)
		if duration:
			print('Consider improving your implementation as this will cause a bottleneck when training.')
		else:
			print('Speed test passed.')

	def task1(self, warmup=False):
		from utils.task1 import compute_recall
		# Load recorded recall and duration for first 100 data points
		recorded_recall = np.load(os.path.join(self.recordings_dir, 'task1_recall.npy'))
		recorded_duration = np.load(os.path.join(self.recordings_dir, 'task1_duration.npy'))
		# Compute recall and duration for the same frame count
		# Displayed on the progress bar are YOUR SOLUTION / RECORDED SOLUTION
		# demo_length = recorded_recall.size
		demo_length = len(self.ds)
		print(demo_length)
		recall = np.empty((demo_length,))
		duration = np.empty((demo_length,))
		pbar = tqdm(range(demo_length), disable=warmup)
		for i in pbar:
			pred, target = self.ds.get_data(i, 'detections'), self.ds.get_data(i, 'target')
			start = timer()
			recall[i] = compute_recall(pred, target, threshold=self.config['eval']['t_rpn_recall'])
			duration[i] = timer() - start
			pbar.set_description('recall [%]: {:.1f} (Target: {:.1f}) | average duration [ms]: {:.1f} (target: {:.1f})'.format(
									recall[:(i+1)].mean()*100, recorded_recall[:(i+1)].mean()*100,
									duration[:(i+1)].mean()*1000, recorded_duration[:(i+1)].mean()*1000))
		# Check results
		if not warmup:
			self.display_test_result(np.abs(recall.mean() - recorded_recall.mean()) <= ERROR_TRESHOLD,
									 duration.mean() > DURATION_THRESHOLD*recorded_duration.mean())

	def task2(self, warmup=False):
		from utils.task2 import roi_pool
		# Load recordings of first data point
		recorded_valid_pred = np.load(os.path.join(self.recordings_dir, 'task2_valid_pred.npy'))
		# ROI pool for first data point
		recorded_duration = np.load(os.path.join(self.recordings_dir, 'task2_duration.npy'))
		start = timer()
		valid_pred, _, _ = roi_pool(pred=self.ds.get_data(task2_idx, 'detections'),
									xyz=self.ds.get_data(task2_idx, 'xyz'),
									feat=self.ds.get_data(task2_idx, 'features'),
									config=self.config['data'])
		duration = timer() - start
		if not warmup:
			print('duration [ms]:  {:.1f} (target: {:.1f} on CPU only bender[59-70])'.format(duration*1000, recorded_duration*1000))
		# Check results
		if not warmup:
			self.display_test_result(
				np.array_equal(valid_pred, recorded_valid_pred),
				duration > DURATION_THRESHOLD*recorded_duration
			)

	def task4(self):
		from utils.task4 import RegressionLoss, ClassificationLoss
		loss = 0
		# Regression loss
		print('Testing regression loss...')
		recorded_loss = 0
		reg_loss = RegressionLoss(self.config['loss'])
		loss = reg_loss(pred=torch.load(os.path.join(self.recordings_dir, 'task4_reg_pred.pt')),
						target=torch.load(os.path.join(self.recordings_dir, 'task4_reg_target.pt')),
						iou=torch.load(os.path.join(self.recordings_dir, 'task4_reg_iou.pt')))
		recorded_loss = torch.load(os.path.join(self.recordings_dir, 'task4_reg_loss.pt'))
		self.display_test_result(loss == recorded_loss)

		# Classification loss
		print('Testing classification loss...')
		cls_loss = ClassificationLoss(self.config['loss'])
		loss = cls_loss(pred=torch.load(os.path.join(self.recordings_dir, 'task4_cls_pred.pt')),
						iou=torch.load(os.path.join(self.recordings_dir, 'task4_cls_iou.pt')))
		recorded_loss = torch.load(os.path.join(self.recordings_dir, 'task4_cls_loss.pt'))
		loss_difference = abs(loss - recorded_loss)
		self.display_test_result(loss_difference < LOSS_THRESHOLD)

	def task5(self):
		from utils.task5 import nms
		# Load input
		pred = np.load(os.path.join(self.recordings_dir, 'task5_pred.npy'))
		score = np.load(os.path.join(self.recordings_dir, 'task5_score.npy'))
		# Load recordings
		recorded_s_f = np.load(os.path.join(self.recordings_dir, 'task5_s_f.npy'))
		recorded_c_f = np.load(os.path.join(self.recordings_dir, 'task5_c_f.npy'))
		# NMS
		s_f, c_f = nms(pred, score, self.config['eval']['t_nms'])

		# Check results
		self.display_test_result(
			(s_f==recorded_s_f).all() and (c_f==recorded_c_f).all()
		)

def replace_username(config):
	user = os.getenv('USER') or os.getenv('USERNAME')  # This works for Linux and Windows
	config['data']['root_dir'] = config['data']['root_dir'].replace('$USER', user)
	config['eval']['output_dir'] = config['eval']['output_dir'].replace('$USER', user)
	config['trainer']['default_root_dir'] = config['trainer']['default_root_dir'].replace('$USER', user)
	return config

if __name__=='__main__':
	config = yaml.safe_load(open(args.config_path, 'r'))
	config = replace_username(config)
	assert(os.path.exists(args.recordings_dir))
	check = CheckTest(config, args.recordings_dir)
	if args.task is None:
		print('You failed the simplest test! Remember to put in your task number with: python tests/test.py --task NUMBER')
		exit()
	if args.task == 3:
		print('There is no test for task 3.')
		exit()
	if args.task in [1, 2]:
		getattr(check, f'task{args.task}')(warmup=True)
		getattr(check, f'task{args.task}')(warmup=True)
		while psutil.cpu_percent(10) > 10:
			print("Wait until the cpu utilization is lower for a comparable runtime benchmark.")
		print("Run actual tests.")
	getattr(check, f'task{args.task}')()
