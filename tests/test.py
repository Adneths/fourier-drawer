import numpy as np
import skvideo.io

import sys
sys.path.insert(1, './../libs/')
from util import printProgressBar

def imgCost(img0, img1):	
	weight = np.maximum(np.max(img0, axis=2), np.max(img1, axis=2))
	return np.sum(np.sum(np.subtract(img0, img1, dtype=np.float64)**2, axis=2) * weight) / (np.sum(weight) * 3 * 255**2)
	
def compareVideo(file0, file1):
	mData0 = skvideo.io.ffprobe(file0)
	mData1 = skvideo.io.ffprobe(file1)
	if mData0['video']['@r_frame_rate'] != mData1['video']['@r_frame_rate']:
		print('{} and {} have different frame rates'.format(file0, file1))
		return -1
	if mData0['video']['@nb_frames'] != mData1['video']['@nb_frames']:
		print('{} and {} have a different number of frames'.format(file0, file1))
		return -1
	
	frames = int(mData0['video']['@nb_frames'])
	
	reader0 = skvideo.io.vreader(file0)
	reader1 = skvideo.io.vreader(file1)
	
	frame = 0
	totalCost = 0;
	for f0, f1 in zip(reader0, reader1):
		frame += 1
		totalCost += imgCost(f0, f1)
		printProgressBar(frame/frames, 'Comparing {} & {}'.format(file0, file1))
	similarity = 1 - totalCost/frame
	print('\nSimilarity: {:.3f}%'.format(100 * similarity))
	
	return similarity


import os
import subprocess
for f in os.listdir('.'):
	if os.path.isdir(f):
		cmd = '(cd {} & render.bat)'.format(f)
		subprocess.call(cmd, shell=True)

count = 0
fail = 0
for f in os.listdir('.'):
	if os.path.isdir(f):
		count += 1
		similarity = compareVideo('{}/sample.mp4'.format(f), '{}/output.mp4'.format(f))
		if similarity < 0.9:
			fail += 1
			print('Error: Failed {}'.format(f))

print('Passed {} out of {} tests'.format(count - fail, count))
		