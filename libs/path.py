import numpy as np
import scipy

from svgpathtools import parse_path
import xml.etree.ElementTree as ET
from PIL import Image
import potrace

import re
import subprocess
import time

from .util import printProgressBar

try:
	import sys
	import os
	dir, file = os.path.split(sys.argv[0])
	cLib = PyDLL(os.path.join(dir,'cpp','lib.dll'),winmode=0)
	cLib.findClosest.argtypes = [c_int, POINTER(c_ulonglong), c_int, POINTER(c_ulonglong)]
	cLib.findClosest.restype = py_object
except:
	print('Warn: Could not load lib.dll')
	cLib = None


def getClosestPair(A, B):
	#Check duplicates
	AB = np.append(np.unique(A), np.unique(B))
	ABq, counts = np.unique(AB, return_counts=True)
	if len(ABq) < len(AB):
		p = ABq[np.argmax(counts > 1)]
		return np.argmax(A == p), np.argmax(B == p), 0
	
	if cLib == None:
		pA = np.transpose([np.real(A),np.imag(A)])
		pB = np.transpose([np.real(B),np.imag(B)])
		M = scipy.spatial.distance.cdist(pA,pB)
		ind = np.unravel_index(np.argmin(M),M.shape)
		return ind[0], ind[1], M[ind[0],ind[1]]
	else:
		print('e')
		return cLib.findClosest(len(A), A.ctypes.data_as(POINTER(c_ulonglong)), len(B), B.ctypes.data_as(POINTER(c_ulonglong)))

def minMax(a,b):
	if a > b:
		return b, a
	return a, b
def mergePaths(paths, showProgress=True):
	if showProgress:
		prog = 0
		total = len(paths)**2
		
	A = np.zeros((len(paths),len(paths),3))
	for i in range(len(paths)):
		for j in range(i+1,len(paths)):
			A[i,j] = getClosestPair(paths[i],paths[j])
			if showProgress:
				prog+=2
				printProgressBar(prog/total, 'Optimizing Path')
	
	span = scipy.sparse.csgraph.minimum_spanning_tree(np.square(A[:,:,2]), overwrite=True)
	if showProgress:
		prog+=1
		printProgressBar(prog/total, 'Optimizing Path')
	
	inds = []
	stack = [0]
	past = [0]
	insert = [0]
	curr = 0
	prev = -1
	n = span.shape[0]
	while len(stack) > 0:
		if len(inds) == 0:
			inds.append([0,0,len(paths[curr])])
		elif not curr in insert:
			insert.append(curr)
			pair = A[minMax(prev,curr)][0:2]
			if prev > curr:
				pair = np.roll(pair,1)
			for i in range(len(inds)):
				if inds[i][0] == prev and inds[i][1] <= pair[0] and inds[i][2] > pair[0]:
					inds.insert(i+1, [prev,pair[0]+1,inds[i][2]])
					inds[i][2] = pair[0]+1
					if pair[1] == 0:
						inds.insert(i+1, [curr,0,len(paths[curr])])
					else:
						inds.insert(i+1, [curr,pair[1],len(paths[curr])])
						inds.insert(i+2, [curr,0,pair[1]])
			if showProgress:
				prog+=1
				printProgressBar(prog/total, 'Optimizing Path')
		
		for i in range(0,n+1):
			if i == n:
				stack.pop()
			elif span[minMax(stack[-1],i)] != 0 and not i in past:
				stack.append(i)
				past.append(i)
				break
		prev = curr
		if len(stack) > 0:
			curr = stack[-1]
	path = np.empty((sum([len(p) for p in paths])), dtype=np.complex128)
	i = 0
	for intv in inds:
		intv = [int(i) for i in intv]
		path[i:i+intv[2]-intv[1]] = paths[intv[0]][intv[1]:intv[2]]
		i += intv[2]-intv[1]
	return path

def pathToPoints(path, density=7, N=-1):
	if N < 1:
		N = int(path.length()*density)
	t = np.linspace(0, len(path), N);
	ts = [t[(t<i)*(t>=i-1)]-i+1 for i in range(1,len(path)+1)]
	ts[-1] = np.append(ts[-1],1)
	x = []
	for seg, param in zip(path,ts):
		x = np.append(x,seg.poly()(param))
	x = np.conjugate(x)
	return x

def get_namespace(element):
	m = re.match('\{.*\}', element.tag)
	if m == None:
		return 0, 0
	return m.group(0) if m else ''
	
def getTranslation(transformStr):
	m = re.match('.*translate\(([0-9\.\-]+),([0-9\.\-]+)\)', transformStr)
	if m == None:
		return 0, 0
	return float(m.group(1)), float(m.group(2))

def getScale(transformStr):
	m = re.match('.*scale\(([0-9\.\-]+),([0-9\.\-]+)\)', transformStr)
	if m == None:
		return 1, 1
	return float(m.group(1)), float(m.group(2))
	
def getArea(bbox):
	return (bbox[1]-bbox[0])*(bbox[3]-bbox[2])

def centerPath(path):
	xMax = max(np.real(path))
	xMin = min(np.real(path))
	yMax = max(np.imag(path))
	yMin = min(np.imag(path))
	return path + ((xMax-xMin)/2-xMax) + 1j*((yMax-yMin)/2-yMax)
	
def boundPath(path, dims):
	w = max(np.real(path))-min(np.real(path))
	h = max(np.imag(path))-min(np.imag(path))
	s = min(dims[0]/w,dims[1]/h)
	return path*s

def svgToPath(file, base_density=7, N=-1):
	tree = ET.parse(file)
	root = tree.getroot()
	namespace = get_namespace(tree.getroot())
	paths = []
	tLen = svgToPathCountLen(root, (0,0), (1,1), namespace)
	svgToPathHelper(paths, root, (0,0), (1,1), tLen, namespace, base_density, N)
	return mergePaths(paths)
def svgToPathCountLen(root, tran, scal, namespace):
	if 'transform' in root.attrib:
		t = getTranslation(root.attrib['transform'])
		tran = (tran[0]+t[0]*scal[0],tran[1]+t[1]*scal[1])
		s = getScale(root.attrib['transform'])
		scal = (scal[0]*s[0],scal[1]*s[1])
	scale = np.sqrt(scal[0]*scal[0]+scal[1]*scal[1])/np.sqrt(2)
	total = 0
	for child in root:
		if child.tag == namespace+'path':
			path = parse_path(child.attrib['d'])
			#Some inaccuracy in length
			total += path.length()*scale
		total += svgToPathCountLen(child, tran, scal, namespace)
	return total
def svgToPathHelper(list, root, tran, scal, tLen, namespace, base_density, N):
	if 'transform' in root.attrib:
		t = getTranslation(root.attrib['transform'])
		tran = (tran[0]+t[0]*scal[0],tran[1]+t[1]*scal[1])
		s = getScale(root.attrib['transform'])
		scal = (scal[0]*s[0],scal[1]*s[1])
	scale = np.sqrt(scal[0]*scal[0]+scal[1]*scal[1])/np.sqrt(2)
	for child in root:
		if child.tag == namespace+'path':
			path = parse_path(child.attrib['d'])
			for p in ([path] if path.iscontinuous() else path.continuous_subpaths()):
				points = pathToPoints(p, density=base_density*scale, N=int(N*p.length()*scale/tLen))
				points = np.real(points)*scal[0] + 1j*np.imag(points)*scal[1]
				points = points + tran[0] - 1j*tran[1]
				list.append(points)
		svgToPathHelper(list, child, tran, scal, tLen, namespace, base_density, N)

def imageFileToPath(file, base_density=7, N=-1):
	return imageToPath(np.asarray(Image.open(file)), base_density, N)

def imageToPath(data, base_density=7, N=-1, showProgress=True):
	if len(data.shape) == 3:
		data = np.sum(data, axis=2)/data.shape[2]
	bmp = potrace.Bitmap(data)
	potPath = bmp.trace()
	pStrs = []
	for curve in potPath:
		prev = 0
		s = ' M {} {} '.format(curve.start_point.x,curve.start_point.y)
		for seg in curve:
			if seg.is_corner:
				if prev != 1:
					s += 'L'
				s += '{} {} {} {} '.format(seg.c.x,seg.c.y,seg.end_point.x,seg.end_point.y)
				prev = 1
			else:
				if prev != 2:
					s += 'C'
				s += '{} {} {} {} {} {} '.format(seg.c1.x,seg.c1.y,seg.c2.x,seg.c2.y,seg.end_point.x,seg.end_point.y)
				prev = 2
		pStrs.append(s+'Z')
	if len(pStrs) == 0:
		R = np.sqrt(data.shape[0]**2+data.shape[1]**2)/2
		scale = 300000/(R**2*np.pi)
		if N < 0:
			N = 2*np.pi*R * base_density*scale
		return np.exp(1j*np.linspace(0,2*np.pi,N))
	else:
		paths = []
		path = parse_path(''.join(pStrs));
		scale = 300000/getArea(path.bbox())
		tLen = path.length()
		for p in ([path] if path.iscontinuous() else path.continuous_subpaths()):
			points = pathToPoints(p, density=base_density*scale, N=int(N*p.length()/tLen))
			paths.append(points)
		return mergePaths(paths,showProgress)

def appendFrames(frames, b):
	np.seterr(divide='ignore')
	m = frames[-1][-1]-frames[-1][-2]
	if m == 0:
		mf = np.zeros((len(b)))
		mb = np.zeros((len(b)))
	else:
		mf = (b-np.roll(b,1))/m
		mb = (b-np.roll(b,-1))/m
		mf = abs(np.imag(np.log(mf)))
		mb = abs(np.imag(np.log(mb)))
	d = abs(b-frames[-1][-1])
	a = np.tile(np.square(d),2) + np.square(np.append(mf,mb))
	i = np.argmin(a)
	if i < len(a)/2:
		frames.append(np.roll(b,-i))
	else:
		frames.append(np.roll(np.flip(b),i-len(b)))

def videoToPath(file, base_density=7, N=-1, dims=None, border=0.9):
	print('Preparing video')
	output = subprocess.check_output('ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format csv \"{}\"'.format(file))
	m = re.search('stream,([0-9]+)', str(output))
	if m != None:
		total = int(m.group(1))
	else:
		total = -1
	reader = skvideo.io.vreader(file)
	frames = []
	
	if total != -1:
		pT = time.time()
		s = 'XX:XX remaining'
		d60 = [0]*60
		tail = 0
		t = time.time()
	
	count = 0
	for frame in reader:
		count+=1
		if dims == None:
			dims = (frame.shape[1],frame.shape[0])
		path = imageToPath(frame, base_density, N, False)
		if len(frames) == 0:
			frames.append(path)
		else:
			appendFrames(frames,path)
		
		if total != -1:
			d60[tail] = time.time()-t
			tail = (tail+1)%60
			if time.time()-pT > 2:
				pT = time.time()
				s = (total-len(frames))*sum(d60)/60
				if s < 3600:
					s = '| {:02}:{:02.0f} remaining         '.format(int((s%3600)/60),s%60)
				else:
					s = '| {}:{:02}:{:02.0f} remaining         '.format(int(s/3600),int((s%3600)/60),s%60)
		
		if total > 0:
			printProgressBar(len(frames)/total, 'Tracing frames', s)
		else:
			print('\rProcessing frame {}'.format(len(frames)), end = '')
		if total != -1:
			t = time.time()
		
	printProgressBar(1, 'Tracing frames', '| 00:00 remaining         ')
	l = sum([len(f) for f in frames])
	a = np.empty((l), dtype=np.complex128)
	i = 0
	for frame in frames:
		a[i:i+len(frame)] = frame
		i += len(frame)
	a = boundPath(a, (dims[0]*border,dims[1]*border))
	return (a, dims, count)