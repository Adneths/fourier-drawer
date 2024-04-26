import numpy as np
import numpy
numpy.float = np.float64
numpy.int = numpy.int_
import scipy
from scipy.optimize import basinhopping

from svgpathtools import parse_path
import xml.etree.ElementTree as ET
from PIL import Image
import potrace

import re
import subprocess
import time
import skvideo.io

from .util import printProgressBar

def sort(a,b):
	if a > b:
		return b, a
	return a, b

def generatePointsAndMergePaths(paths, showProgress=True):
	prog = 0
	total = len(paths)*(1+1+2) + len(paths)*(len(paths)-1)//2
	
	if showProgress:
		print('\r({}/{})'.format(prog,total), end='')
	points = [[]] * len(paths)
	for i in range(len(paths)):
		points[i] = pathToPoints(*paths[i])
		prog += 1
		if showProgress:
			printProgressBar(prog/total, 'Processing Path')
	
	data = []
	trees = []
	for i in range(len(points)):
		data.append(np.stack([np.real(points[i]),np.imag(points[i])],axis=1))
		trees.append(scipy.spatial.KDTree(data[-1]))
		prog += 1
		if showProgress:
			printProgressBar(prog/total, 'Processing Path')
	
	D = np.zeros((len(paths),len(paths),3))
	for i in range(len(paths)):
		for j in range(i+1,len(paths)):
			if len(points[j]) < len(points[i]):
				res = trees[i].query(data[j])
				ind = np.argmin(res[0])
				D[i,j] = (res[0][ind], res[1][ind], ind)
			else:
				res = trees[j].query(data[i])
				ind = np.argmin(res[0])
				D[i,j] = (res[0][ind], ind, res[1][ind])
			prog += 1
			if showProgress:
				printProgressBar(prog/total, 'Processing Path')
	span = scipy.sparse.csgraph.minimum_spanning_tree(D[:,:,0], overwrite=True)
	span = [i for i in zip(*span.nonzero())]
	
	test = []
	intervals = [(0,0,len(points[0]))] # pathId, indStart, indEnd
	intervalsPtr = [[] for i in range(len(paths))] # intervalsInd, indStart, indEnd
	intervalsPtr[0].append((0,0,len(points[0])))
	def insert(p, q):
		pairInd = (int(D[p,q][1]),int(D[p,q][2])) if p < q else (int(D[q,p][2]),int(D[q,p][1]))
		for pt in range(len(intervalsPtr[p])):
			intv = intervalsPtr[p][pt]
			if pairInd[0] >= intv[1] and pairInd[0] < intv[2]:
				intervals.insert(intv[0]+1, (p, pairInd[0], intv[2]))
				intervals.insert(intv[0]+1, (q, 0, pairInd[1]))
				intervals.insert(intv[0]+1, (q, pairInd[1], len(points[q])))
				intervals[intv[0]] = (p, intv[1], pairInd[0])
				
				
				for ptrs in intervalsPtr:
					for i in range(len(ptrs)):
						if ptrs[i][0] > intv[0]:
							ptrs[i] = (ptrs[i][0]+3, ptrs[i][1], ptrs[i][2])
				
				intervalsPtr[q].append((intv[0]+2, 0, pairInd[1]))
				intervalsPtr[q].append((intv[0]+1, pairInd[1], len(points[q])))
				intervalsPtr[p].insert(pt+1, (intv[0]+3, pairInd[0], intv[2]))
				intervalsPtr[p][pt] = (intv[0], intv[1], pairInd[0])
				break
	stack = [0]
	past = [0]
	while len(stack) > 0:
		curr = stack.pop()
		for pair in span:
			if pair[0] == curr and not pair[1] in past:
				insert(curr, pair[1])
				stack.append(pair[1])
				past.append(pair[1])
				prog += 2
				if showProgress:
					printProgressBar(prog/total, 'Processing Path')
			if pair[1] == curr and not pair[0] in past:
				insert(curr, pair[0])
				stack.append(pair[0])
				past.append(pair[0])
				prog += 2
				if showProgress:
					printProgressBar(prog/total, 'Processing Path')
	
	path = np.empty(np.sum([len(pts) for pts in points]), dtype=np.complex128)
	ind = 0
	for i in range(len(intervalsPtr)):
		l = 0
		for p in intervalsPtr[i]:
			l += p[2] - p[1]
	
	for intv in intervals:
		path[ind:ind+intv[2]-intv[1]] = points[intv[0]][intv[1]:intv[2]]
		ind += intv[2]-intv[1]
		
	prog += 2
	if showProgress:
		printProgressBar(prog/total, 'Processing Path')
		print()
	
	return path

def pathToPoints(path, density=7, N=-1):
	if N < 0:
		N = int(path.length()*density)
	elif N == 0:
		return []
		
	path._calc_lengths()
	lengths = np.cumsum(np.append([0],path._lengths))
	t = np.linspace(0, 1, N)
	ts = [t[np.logical_and(t>lengths[i],t<=lengths[i+1])] for i in range(len(lengths)-1)]
	if N > 1:
		ts[0] = np.append([0], ts[0])
	for i in range(len(ts)):
		ts[i] = (ts[i] - lengths[i]) / (lengths[i+1] - lengths[i])
	x = []
	for seg, param in zip(path,ts):
		x = np.append(x,seg.points(param))
	x = np.conjugate(x)
	return x

def get_namespace(element):
	m = re.match('\\{.*\\}', element.tag)
	if m is None:
		return 0, 0
	return m.group(0) if m else ''
	
def getTranslation(transformStr):
	m = re.match('.*translate\\(([0-9\\.\\-]+),([0-9\\.\\-]+)\\)', transformStr)
	if m is None:
		return 0, 0
	return float(m.group(1)), float(m.group(2))

def getScale(transformStr):
	m = re.match('.*scale\\(([0-9\\.\\-]+),([0-9\\.\\-]+)\\)', transformStr)
	if m is None:
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

def svgToPath(file, base_density=7, N=-1, tosave=False):
	tree = ET.parse(file)
	root = tree.getroot()
	namespace = get_namespace(tree.getroot())
	path_factors = []
	tLen = svgToPathCountLen(root, (0,0), (1,1), namespace)
	svgToPathHelper(path_factors, root, (0,0), (1,1), tLen, namespace, base_density, N, [] if tosave else None)
	return generatePointsAndMergePaths(path_factors), raw_path_factors
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
			total += path.length()*scale
		total += svgToPathCountLen(child, tran, scal, namespace)
	return total
def svgToPathHelper(path_factors, root, tran, scal, tLen, namespace, base_density, N, raw_path_factor=None):
	if 'transform' in root.attrib:
		t = getTranslation(root.attrib['transform'])
		tran = (tran[0]+t[0]*scal[0],tran[1]+t[1]*scal[1])
		s = getScale(root.attrib['transform'])
		scal = (scal[0]*s[0],scal[1]*s[1])
	scale = np.sqrt(scal[0]*scal[0]+scal[1]*scal[1])/np.sqrt(2)
	dP = tLen/N
	rLen = 0
	for child in root:
		if child.tag == namespace+'path':
			path = parse_path(child.attrib['d'])
			for p in ([path] if path.iscontinuous() else path.continuous_subpaths()):
				path_factors.append((p, base_density*scale, -1 if N == -1 else int(N*(p.length()*scale+rLen%dP)/tLen)))
				rLen += p.length()*scale
			if raw_path_factor != None:
				raw_path_factor.append((child.attrib['d'], scale, tLen))
		svgToPathHelper(path_factors, child, tran, scal, tLen, namespace, base_density, N, raw_path_factor=raw_path_factor)

def imageFileToPath(file, base_density=7, N=-1, tosave=False):
	return imageToPath(np.asarray(Image.open(file)), base_density, N, tosave=tosave)
def imageToPath(data, base_density=7, N=-1, tosave=False, showProgress=True):
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
		'''
		R = np.sqrt(data.shape[0]**2+data.shape[1]**2)/2
		scale = 300000/(R**2*np.pi)
		if N < 0:
			N = 2*np.pi*R * base_density*scale
		return np.exp(1j*np.linspace(0,2*np.pi,N))
		'''
		return None, None
	else:
		path_factors = []
		path = parse_path(''.join(pStrs))
		scale = 300000/getArea(path.bbox())
		rLen = 0
		tLen = path.length()
		dP = tLen/N
		raw_path_factor = [(''.join(pStrs), scale, tLen)] if tosave else None
		for p in ([path] if path.iscontinuous() else path.continuous_subpaths()):
			path_factors.append((p, base_density*scale, -1 if N == -1 else int(N*(p.length()+rLen%dP)/tLen)))
			rLen += p.length()
		return generatePointsAndMergePaths(path_factors, showProgress), raw_path_factor

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

def videoToPath(file, base_density=7, N=-1, dims=None, border=0.9, tosave=False):
	print('Preparing video')
	output = subprocess.check_output('ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format csv \"{}\"'.format(file))
	m = re.search('stream,([0-9]+)', str(output))
	if m != None:
		total = int(m.group(1))
	else:
		total = -1
	reader = skvideo.io.vreader(file)
	frames = []
	raw_path_factors = [] if tosave else None
	
	if total != -1:
		pT = time.time()
		s = 'XX:XX remaining'
		d60 = [0]*60
		tail = 0
		t = time.time()
	
	count = 0
	prev_path = np.zeros((0))
	for frame in reader:
		count+=1
		if dims is None:
			dims = (frame.shape[1],frame.shape[0])
		path, raw_path = imageToPath(frame, base_density, N, tosave=tosave, showProgress=False)
		if len(frames) == 0:
			frames.append(prev_path if path==None else path.copy())
		else:
			appendFrames(frames, prev_path if path==None else path)
		if not path is None:
			prev_path = path
		if tosave:
			raw_path_factors.append(raw_path)
		
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
	
	prog = 0
	total = len(frames)
	l = sum([len(f) for f in frames])
	a = np.empty((l), dtype=np.complex128)
	i = 0
	for frame in frames:
		printProgressBar(prog/total, 'Merging frames', '|                           ')
		prog += 1
		a[i:i+len(frame)] = frame
		i += len(frame)
	printProgressBar(1, 'Merging frames', '|                           ')
	print()
	#a = boundPath(a, (dims[0],dims[1]))
	return a, dims, count, raw_path_factors

def resamplePath(raw_path, density=7, N=-1, types=(True, False, False, False), showProgress=True):
	print('Resampling Path')
	frames = []
	prev_path = None
	for idx, frame in enumerate(raw_path):
		if frame is None and prev_path is None:
			print('Unable to resample frame {}, skipping'.format(idx))
			continue
		if not frame is None:
			path_factors = []
			for raw_path_factor in frame:
				path = parse_path(raw_path_factor[0])
				scale = raw_path_factor[1]
				tLen = raw_path_factor[2]
				rLen = 0
				dP = tLen/N
				for p in ([path] if path.iscontinuous() else path.continuous_subpaths()):
					path_factors.append((p, base_density*scale, -1 if N == -1 else int(N*(p.length()+rLen%dP)/tLen)))
					rLen += p.length() * scale if types[0] else p.length()
			path = generatePointsAndMergePaths(path_factors, showProgress and not types[3])
		else
			path = None
		
		if len(frames) == 0:
			frames.append(prev_path if path==None else path.copy())
		else:
			appendFrames(frames, prev_path if path==None else path)
		if not path is None:
			prev_path = path
	
	if len(frames) > 1:
		prog = 0
		total = len(frames)
		l = sum([len(f) for f in frames])
		a = np.empty((l), dtype=np.complex128)
		i = 0
		for frame in frames:
			printProgressBar(prog/total, 'Merging frames', '|                           ')
			prog += 1
			a[i:i+len(frame)] = frame
			i += len(frame)
		printProgressBar(1, 'Merging frames', '|                           ')
		print()
		return a
	else
		return frames[0]
	