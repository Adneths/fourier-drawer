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
'''
def dist(path1, path2):
	#print(path1, path2)
	def dist(t):
		print(path1.radialrange(path2.point(t))[0][0])
		return path1.radialrange(path2.point(t))[0][0]
	T2 = fminbound(dist, 0, 1)
	pt2 = path2.point(T2)
	print('Result:', T2, pt2)
	T1Pair = path1.radialrange(pt2)[0][1:3]
	T1 = path1.t2T(T1Pair[1],T1Pair[0])
	pt1 = path1.point(T1)
	print('       ', T1, pt1)
	from svgpathtools import disvg, Line
	disvg([path1, path2, Line(pt1, pt2)], nodes=[pt1, pt2])
	#exit()
	print()
	return (abs(pt1-pt2), T1, T2)
'''
'''
def minimizeDistanceForSegments(seg1, seg2):
	def dist(t):
		return seg1.radialrange(seg2.point(t))[0][0]
	T2 = fminbound(dist, 0, 1)
	pt2 = seg2.point(T2)
	T1 = seg1.radialrange(pt2)[0][1]
	pt1 = seg1.point(T1)
	return (abs(pt1-pt2), T1, T2)
def minimizeDistanceForPathToSegment(path1, seg2):
	def dist(t):
		return path1.radialrange(seg2.point(t))[0][0]
	T2 = fminbound(dist, 0, 1)
	pt2 = seg2.point(T2)
	T1 = path1.radialrange(pt2)[0][1]
	pt1 = path1.point(T1)
	return (abs(pt1-pt2), T1, T2)
def minimizeDistanceForPaths(path1, path2):
	global_min = (np.inf, (0,0), (0,0))
	n = len(path1) * len(path2)
	i = 0
	for id1, seg1 in enumerate(path1):
		for id2, seg2 in enumerate(path2):
			i += 1
			print('({}/{})'.format(i,n))
			local_min = minimizeDistanceForSegments(seg1, seg2)
			if global_min[0] > local_min[0]:
				global_min = (local_min[0], (id1, local_min[1]), (id2, local_min[2]))
	return (global_min[0], path1.t2T(*global_min[1]), path2.t2T(*global_min[2]))
'''
def binsearch(arr, val):
	start = 0
	end = len(arr)
	while start < end-1:
		mid = (start+end)//2
		if arr[mid] < val:
			start = mid
		else:
			end = mid
	return start-1
def minimizeDistanceForPaths(path1, path2):
	path2._calc_lengths()
	path2L = np.cumsum(np.append([0],path2._lengths))
	def dist(t):
		#ind = binsearch(path2L, t)
		#return path1.radialrange(path2[ind].point( (t-path2L[ind]) / path2._lengths[ind] ))[0][0]
		return path1.radialrange(path2.point(t))[0][0]
	T2 = basinhopping(dist, 0.5, niter=100, T=0.25, stepsize=0.5, niter_success=15, minimizer_kwargs={'bounds': [(0,1)], 'options': {'maxiter': 10}}).x[0]
	#T2 = scipy.optimize.fminbound(dist, 0, 1)
	pt2 = path2.point(T2)
	T1Pair = path1.radialrange(pt2)[0][1:3]
	T1 = path1.t2T(T1Pair[1],T1Pair[0])
	pt1 = path1.point(T1)
	#print('Result:', T2, pt2)
	#print('       ', T1, pt1)
	#from svgpathtools import disvg, Line
	#disvg([path1, path2, Line(pt1, pt2)], nodes=[pt1, pt2])
	##print(pt1, pt2, end=' : ')
	return (abs(pt1-pt2), T1, T2)
def minimizeDistanceForPoints(points1, points2):
	def dist(idxs):
		return abs(points1[int(idxs[0])] - points2[int(idxs[1])])
	idx1, idx2 = basinhopping(dist, (len(points1)//2, len(points2)//2), niter=200, T=(len(points1)+len(points2))/40, stepsize=(len(points1)+len(points2))/4, niter_success=75, minimizer_kwargs={'bounds': [(0,len(points1)-1),(0,len(points2)-1)], 'options': {'maxiter': 300} }, seed=1).x
	return (dist((idx1, idx2)), int(idx1), int(idx2))
def mergePaths(paths, showProgress=True):
	# From points
	'''
	points = [[]] * len(paths)
	for i in range(len(paths)):
		points[i] = pathToPoints(*paths[i])
	D = np.zeros((len(paths),len(paths),3))
	prog = 0;
	total = len(paths) * (len(paths)-1) // 2
	print('\r({}/{})'.format(prog,total), end='')
	for i in range(len(paths)):
		for j in range(i+1,len(paths)):
			D[i,j] = minimizeDistanceForPoints(points[i], points[j])
			prog += 1
			print('\r({}/{})'.format(prog,total), end='')
	print()
	span = scipy.sparse.csgraph.minimum_spanning_tree(D[:,:,0], overwrite=True)
	span = [i for i in zip(*span.nonzero())]
	
	test = []
	node = []
	from svgpathtools import disvg, Line
	for p in paths:
		test.append(p[0])
	for pair in span:
		pPair = sort(*pair)
		tPair = D[*pair][1:3]
		pt1 = np.conjugate(points[pPair[0]][int(tPair[0])])
		node.append(pt1)
		test.append(pt1)
		pt2 = np.conjugate(points[pPair[1]][int(tPair[1])])
		node.append(pt2)
		test.append(pt2)
		test.append(Line(pt1, pt2))
	disvg(test, nodes=node)
	exit()
	
	intervals = [(0,0,len(points[0]))] # pathId, indStart, indEnd
	intervalsPtr = [[] for i in range(len(paths))] # intervalsInd, indStart, indEnd
	intervalsPtr[0].append((0,0,len(points[0])))
	def insert(p, q):
		pairInd = (int(D[p,q][1]),int(D[p,q][2])) if p < q else (int(D[p,q][2]),int(D[p,q][1]))
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
			if pair[1] == curr and not pair[0] in past:
				insert(curr, pair[0])
				stack.append(pair[0])
				past.append(pair[0])
	
	path = np.empty(np.sum([len(pts) for pts in points]), dtype=np.complex128)
	ind = 0
	for i in range(len(intervalsPtr)):
		l = 0
		for p in intervalsPtr[i]:
			l += p[2] - p[1]
	for intv in intervals:
		path[ind:ind+intv[2]-intv[1]] = points[intv[0]][intv[1]:intv[2]]
		ind += intv[2]-intv[1]
	return path
	'''

	# From paths
	#'''
	points = [[]] * len(paths)
	##test = [[]] * len(paths)
	##test2 = []
	for i in range(len(paths)):
		points[i] = pathToPoints(*paths[i])
	D = np.zeros((len(paths),len(paths),3))
	prog = 0;
	total = len(paths) * (len(paths)-1) // 2
	# if showProgress:
	print('\r({}/{})'.format(prog,total), end='')
	for i in range(len(paths)):
		for j in range(i+1,len(paths)):
			##print(i,j, end=' : ')
			D[i,j] = minimizeDistanceForPaths(paths[i][0], paths[j][0])
			##print(points[i][int(D[i,j][1]*len(points[i]))], points[j][int(D[i,j][2]*len(points[j]))], end=' : ')
			##print(np.conjugate(paths[i][0].point(D[i,j][1])), np.conjugate(paths[j][0].point(D[i,j][2])))
			##test2.append(np.conjugate(paths[i][0].point(D[i,j][1])))
			##test2.append(np.conjugate(paths[j][0].point(D[i,j][2])))
			prog += 1
			# if showProgress:
			print('\r({}/{})'.format(prog,total), end='')
	if showProgress:
		print()
	span = scipy.sparse.csgraph.minimum_spanning_tree(D[:,:,0], overwrite=True)
	span = [i for i in zip(*span.nonzero())]
	
	###test = []
	###node = []
	###from svgpathtools import disvg, Line
	###for p in paths:
	###	test.append(p[0])
	###for pair in span:
	###	pPair = minMax(*pair)
	###	tPair = D[*pair][1:3]
	###	pt1 = paths[pPair[0]][0].point(tPair[0])
	###	node.append(pt1)
	###	test.append(pt1)
	###	pt2 = paths[pPair[1]][0].point(tPair[1])
	###	node.append(pt2)
	###	test.append(pt2)
	###	test.append(Line(pt1, pt2))
	###disvg(test, nodes=node)
	
	###test = []
	intervals = [(0,0,len(points[0]))] # pathId, indStart, indEnd
	intervalsPtr = [[] for i in range(len(paths))] # intervalsInd, indStart, indEnd
	intervalsPtr[0].append((0,0,len(points[0])))
	def insert(p, q):
		pairInd = (int(D[p,q][1]*len(points[p])),int(D[p,q][2]*len(points[q]))) if p < q else (int(D[q,p][2]*len(points[p])),int(D[q,p][1]*len(points[q])))
		###test.append(points[p][pairInd[0]])
		###test.append(points[q][pairInd[1]])
		##print(p,q, pairInd)
		##print('Preinsert')
		##print('Intervals:', intervals)
		##print('Intveral p:', intervalsPtr[p])
		##print('Interval q:', intervalsPtr[q])
		for pt in range(len(intervalsPtr[p])):
			intv = intervalsPtr[p][pt]
			if pairInd[0] >= intv[1] and pairInd[0] < intv[2]:
				###print(intv)
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
		##print('Postinsert')
		##print('Intervals:', intervals)
		##print('Intveral p:', intervalsPtr[p])
		##print('Interval q:', intervalsPtr[q])
		##print()
	stack = [0]
	past = [0]
	while len(stack) > 0:
		curr = stack.pop()
		for pair in span:
			if pair[0] == curr and not pair[1] in past:
				insert(curr, pair[1])
				stack.append(pair[1])
				past.append(pair[1])
			if pair[1] == curr and not pair[0] in past:
				insert(curr, pair[0])
				stack.append(pair[0])
				past.append(pair[0])
	#print(span)
	#print(intervals)
	#print(intervalsPtr)
	#print([len(pts) for pts in points])
	path = np.empty(np.sum([len(pts) for pts in points]), dtype=np.complex128)
	ind = 0
	for i in range(len(intervalsPtr)):
		l = 0
		for p in intervalsPtr[i]:
			l += p[2] - p[1]
		#print(l, len(points[i]))
	
	for intv in intervals:
		#print(intv, ind, intv[2]-intv[1], ind+intv[2]-intv[1], len(path))
		path[ind:ind+intv[2]-intv[1]] = points[intv[0]][intv[1]:intv[2]]
		ind += intv[2]-intv[1]
		
	###import matplotlib.pyplot as plt
	###plt.plot(np.real(path),np.imag(path),linewidth=0.2)
	###plt.scatter(np.real(test), np.imag(test), s=2)
	##plt.scatter(np.real(test2), np.imag(test2), s=2, c='r')
	###plt.show()
	###exit()
	return path
	#'''

	'''
	if showProgress:
		prog = 0
		total = len(paths)*(len(paths)+5)/2
	
	data = []
	trees = []
	for i in range(len(paths)):
		data.append(np.stack([np.real(paths[i]),np.imag(paths[i])],axis=1))
		trees.append(scipy.spatial.KDTree(data[-1]))
		if showProgress:
			prog+=2
			printProgressBar(prog/total, 'Optimizing Path')
	
	A = np.zeros((len(paths),len(paths),3))
	for i in range(len(paths)):
		for j in range(i+1,len(paths)):
			res = trees[i].query(data[j])
			ind = np.argmin(res[0])
			A[i,j] = (res[1][ind], ind, res[0][ind])
			if showProgress:
				prog+=1
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
	'''

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
		#ts[-1] = np.append(ts[-1],1)
		ts[0] = np.append([0], ts[0])
	for i in range(len(ts)):
		ts[i] = (ts[i] - lengths[i]) / (lengths[i+1] - lengths[i])
	x = []
	for seg, param in zip(path,ts):
		x = np.append(x,seg.points(param))
	x = np.conjugate(x)
	return x
	'''
	if N < 0:
		N = int(path.length()*density)
	elif N == 0:
		return []
	t = np.linspace(0, len(path), N)
	ts = [t[(t<i)*(t>=i-1)]-i+1 for i in range(1,len(path)+1)]
	if N > 1:
		ts[-1] = np.append(ts[-1],1)
	x = []
	for seg, param in zip(path,ts):
		x = np.append(x,seg.poly()(param))
	x = np.conjugate(x)
	return x
	'''

def get_namespace(element):
	m = re.match('\\{.*\\}', element.tag)
	if m == None:
		return 0, 0
	return m.group(0) if m else ''
	
def getTranslation(transformStr):
	m = re.match('.*translate\\(([0-9\\.\\-]+),([0-9\\.\\-]+)\\)', transformStr)
	if m == None:
		return 0, 0
	return float(m.group(1)), float(m.group(2))

def getScale(transformStr):
	m = re.match('.*scale\\(([0-9\\.\\-]+),([0-9\\.\\-]+)\\)', transformStr)
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
			total += path.length()*scale
		total += svgToPathCountLen(child, tran, scal, namespace)
	return total
def svgToPathHelper(paths, root, tran, scal, tLen, namespace, base_density, N):
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
				paths.append((p, base_density*scale, -1 if N == -1 else int(N*(p.length()*scale+rLen%dP)/tLen)))
				'''
				points = pathToPoints(p, density=base_density*scale, N=-1 if N == -1 else int(N*(p.length()*scale+rLen%dP)/tLen))
				rLen += p.length()*scale
				points = np.real(points)*scal[0] + 1j*np.imag(points)*scal[1]
				points = points + tran[0] - 1j*tran[1]
				if len(points) > 0:
					paths.append(points)
				'''
		svgToPathHelper(paths, child, tran, scal, tLen, namespace, base_density, N)

def imageFileToPath(file, base_density=7, N=-1):
	return imageToPath(np.asarray(Image.open(file)), base_density, N)

import time
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
		rLen = 0
		tLen = path.length()
		dP = tLen/N
		for p in ([path] if path.iscontinuous() else path.continuous_subpaths()):
			paths.append((p, base_density*scale, -1 if N == -1 else int(N*(p.length()*scale+rLen%dP)/tLen)))
			'''
			points = pathToPoints(p, density=base_density*scale, N=-1 if N == -1 else int(N*(p.length()+rLen%dP)/tLen))
			rLen += p.length()
			if len(points) > 0:
				paths.append(points)
			'''
		start = time.perf_counter()
		temp = mergePaths(paths,showProgress)
		print(time.perf_counter() - start)
		return temp

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
	a = boundPath(a, (dims[0],dims[1]))
	return (a, dims, count)