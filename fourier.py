import pyglet
from pyglet.gl import *
from pyglet import clock
from ctypes import *

import numpy as np
import numexpr as ne
import scipy
from collections import deque
from svgpathtools import parse_path
import xml.etree.ElementTree as ET
from PIL import Image
import skvideo.io
import re
import potrace
from PIL import Image
import subprocess
import time

def printProgressBar(percentage, prefix = '', suffix='', decimals = 1, length = 40):
	percent = ("{0:." + str(decimals) + "f}").format(100*percentage)
	fill = int(length * percentage)
	bar = '█' * fill + '-' * (length - fill)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = '')

class World(object):
	CYCLE_DURATION = 2*np.pi
	CYCLE_DURATION_EXT = 2.1*np.pi

	def __init__(self, weights, freqs, dims=(800,800)):
		window = pyglet.window.Window(dims[0], dims[1])
		window.minimize()
		
		self.dims = dims
		self.weights, self.freqs = zip(*sorted(zip(weights,freqs), key=lambda pair: -abs(pair[0])))
		self.weights = np.asarray(self.weights)
		self.freqs = np.asarray(self.freqs)
		
		
		self.center = (dims[0]/2) + (dims[1]/2)*1j
		self.time = 0
		self.cutIndex = 0
		
		self.setVectorColor(0xffffff)
		self.setPathColor(0xffff00)
		self.setPathFade(True)
		self.configTime(1)
		self.configTrail(self.CYCLE_DURATION_EXT)
		
		glEnableClientState(GL_VERTEX_ARRAY)
		self.fbo = GLuint(0)
		glGenFramebuffers(1, pointer(self.fbo))
		
		colorBuffer = GLuint(0)
		glGenRenderbuffers(1, pointer(colorBuffer))
		
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		glBindRenderbuffer(GL_RENDERBUFFER, colorBuffer)
		glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, dims[0], dims[1])
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorBuffer)
		
		self.writer = None
	
	def configTime(self, timescale):
		self.timescale = timescale
		
	def configTrail(self, trailDuration):
		self.trailDuration = trailDuration
		
	def setVectorColor(self, hexCode):
		self.vecColor = (((hexCode>>16)&0xff)/255,((hexCode>>8)&0xff)/255,((hexCode)&0xff)/255)
		
	def setPathColor(self, hexCode):
		self.pathColor = (((hexCode>>16)&0xff)/255,((hexCode>>8)&0xff)/255,((hexCode)&0xff)/255)
		
	def setPathFade(self, fade):
		self.pathFade = fade
	
	
	def generateTrail(self):
		self.path = np.full((int(self.trailDuration*60/self.timescale)), np.sum(np.append(self.center, self.weights * ne.evaluate('exp(1j*k*t)', local_dict = {'k': self.freqs, 't': self.time}))), dtype=np.complex128)
		self.tail = 0
		self.generatePathColorArray()
	
	def generatePathColorArray(self):
		self.pathColorArr = np.tile([self.pathColor[0],self.pathColor[1],self.pathColor[2],0],self.path.size)
		self.pathColorArr[3::4] = np.linspace(0,1,self.path.size)
		self.pathColorArr = (GLfloat * self.pathColorArr.size)(*self.pathColorArr)
	
	
	def render(self, duration=CYCLE_DURATION, fps=60, fpf=1, output = 'out'):
		self.generateTrail()
		self.vecs = np.append(self.center, self.weights)
		self.step = np.append(1, ne.evaluate('exp(1j*k*t)', local_dict = {'k': self.freqs, 't': self.dt()}))
		
		def render():
			frame = 0
			
			pT = time.time()
			s = 'XX:XX remaining'
			dL = max(100,fpf)
			d100 = [0]*dL
			tail = 0
			
			self.writer = skvideo.io.FFmpegWriter('{}.mp4'.format(output), inputdict={'-r': str(fps)}, outputdict={'-vcodec': 'libx264', '-vf': 'format=yuv420p'})
			while self.time < duration:
				t = time.time()
				self.draw(frame==0)
				
				d100[tail] = time.time()-t
				tail = (tail+1)%dL
				frame = (frame+1)%fpf
				if time.time()-pT > 2:
					pT = time.time()
					s = (duration-self.time)*60/self.timescale*sum(d100)/dL
					if s < 3600:
						s = '| {:02}:{:02.0f} remaining'.format(int((s%3600)/60),s%60)
					else:
						s = '| {}:{:02}:{:02.0f} remaining'.format(int(s/3600),int((s%3600)/60),s%60)
				printProgressBar(self.time/duration, 'Rendering', s)
			printProgressBar(1, 'Rendering', '00:00 remaining          ')
			exit()
		clock.schedule(lambda dt: render())
		pyglet.app.run()
	
	def dt(self):
		return 1/60 * self.timescale
	
	
	def display(self):
		glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
		glClear(GL_COLOR_BUFFER_BIT)
		glBlitFramebuffer(0, 0, self.dims[0], self.dims[1], 0, 0, self.dims[0], self.dims[1], GL_COLOR_BUFFER_BIT, GL_NEAREST)
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
	
	def draw(self, save):
		self.time += self.dt()
		
		if len(self.step) < 300000:
			self.vecs = self.vecs * self.step
		else:
			self.vecs = ne.evaluate('a*b', local_dict = {'a': self.vecs, 'b': self.step})
			
		if not save:
			self.path[self.tail] = ne.evaluate('sum(v)', local_dict = {'v': self.vecs})
			self.tail = (self.tail+1)%len(self.path)
		else:
			vecSum = np.cumsum(self.vecs)
			self.path[self.tail] = vecSum[-1]
			self.tail = (self.tail+1)%len(self.path)
			
			cutIndex = np.argmax(abs(vecSum-vecSum[-1])<1)
			if self.cutIndex < cutIndex:
				self.cutIndex = cutIndex
			vecSum = vecSum[0:self.cutIndex]
		
		
			glClear(GL_COLOR_BUFFER_BIT)
			glLoadIdentity()
			
			glColor3f(self.vecColor[0],self.vecColor[1],self.vecColor[2])
			glLineWidth(1)
			array = np.empty((vecSum.size*2))
			array[0::2] = np.real(vecSum)
			array[1::2] = np.imag(vecSum)
			data = (GLfloat * array.size)(*array)
			glVertexPointer(2, GL_FLOAT, 0, data)
			glDrawArrays(GL_LINE_STRIP, 0, vecSum.size)
			
			glLineWidth(1.5)
			p = np.roll(self.path,-self.tail)
			array = np.empty((p.size*2))
			array[0::2] = np.real(p)
			array[1::2] = np.imag(p)
			data = (GLfloat * array.size)(*array)
			glEnable(GL_BLEND)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			if self.pathFade:
				glEnableClientState(GL_COLOR_ARRAY)
				glColorPointer(4, GL_FLOAT, 0, self.pathColorArr)
			else:
				glColor3f(self.pathColor[0],self.pathColor[1],self.pathColor[2])
			glVertexPointer(2, GL_FLOAT, 0, data)
			glDrawArrays(GL_LINE_STRIP, 0, p.size)
			glDisableClientState(GL_COLOR_ARRAY)
			glDisable(GL_BLEND)
			
			if self.writer != None:
				data = (GLubyte * (self.dims[0] * self.dims[1] * len('RGBA')))()
				glReadPixels(0, 0, self.dims[0], self.dims[1], GL_RGBA, GL_UNSIGNED_BYTE, pointer(data))
				im = Image.frombytes('RGBA', self.dims, data)
				arr = np.flip(np.asarray(im),0)
				self.writer.writeFrame(arr)


def getClosestPair(A, B):
	#Check duplicates
	AB = np.append(np.unique(A), np.unique(B))
	ABq, counts = np.unique(AB, return_counts=True)
	if len(ABq) < len(AB):
		p = ABq[np.argmax(counts > 1)]
		return np.argmax(A == p), np.argmax(B == p), 0
	
	pA = np.transpose([np.real(A),np.imag(A)])
	pB = np.transpose([np.real(B),np.imag(B)])
	M = scipy.spatial.distance.cdist(pA,pB)
	ind = np.unravel_index(np.argmin(M),M.shape)
	return ind[0], ind[1], M[ind[0],ind[1]]

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
	


def boundPath(path, dims):
	w = max(np.real(path))-min(np.real(path))
	h = max(np.imag(path))-min(np.imag(path))
	s = min(dims[0]/w,dims[1]/h)
	return path*s

def renderPath(path, dims, duration, timescale, trailLength, trailFade, trailColor, vectorColor, fps, fpf, output):
	N = len(path)
	X = np.fft.fft(path)
	freqs = np.append(np.arange(0,int(N/2)),np.arange(-int(np.ceil(N/2)),0))
	
	world = World(X/N,freqs,dims)
	world.configTime(timescale)
	world.configTrail(trailLength)
	world.setPathFade(trailFade)
	world.setPathColor(trailColor)
	world.setVectorColor(vectorColor)
	world.render(fps=fps,duration=duration,fpf=fpf,output=output)

def imageFileToPath(file, base_density=7, N=-1):
	return imageToPath(np.asarray(Image.open(file)), base_density, N)

def imageToPath(data, base_density=7, N=-1, showProgress=True):
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
		R = np.sqrt(data.size[0]**2+data.size[1]**2)/2
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
					s = '| {:02}:{:02.0f} remaining'.format(int((s%3600)/60),s%60)
				else:
					s = '| {}:{:02}:{:02.0f} remaining'.format(int(s/3600),int((s%3600)/60),s%60)
		
		if total > 0:
			printProgressBar(len(frames)/total, 'Tracing frames', s)
		else:
			print('\rProcessing frame {}'.format(len(frames)), end = '')
		if total != -1:
			t = time.time()
		
	printProgressBar(1, 'Tracing frames', '| 00:00 remaining')
	l = sum([len(f) for f in frames])
	a = np.empty((l), dtype=np.complex128)
	i = 0
	for frame in frames:
		a[i:i+len(frame)] = frame
		i += len(frame)
	a = boundPath(a, (dims[0]*border,dims[1]*border))
	return (a, dims, count)