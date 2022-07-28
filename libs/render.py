from ctypes import *
import skvideo.io

import numpy as np
import numexpr as ne
import scipy
from PIL import Image

import glfw
from OpenGL.GL import *

import time

from .util import printProgressBar

class World(object):
	CYCLE_DURATION = 2*np.pi
	CYCLE_DURATION_EXT = 2.1*np.pi

	def __init__(self, weights, freqs, dims=(800,800)):
		self.dims = dims
		self.weights, self.freqs = zip(*sorted(zip(weights,freqs), key=lambda pair: -abs(pair[0])))
		self.weights = np.asarray(self.weights)
		self.freqs = np.asarray(self.freqs)
		
		self.time = 0
		self.cutIndex = 0
		
		self.setVectorColor(0xffffff)
		self.setPathColor(0xffff00)
		self.setPathFade(True)
		self.configTime(1)
		self.configTrail(self.CYCLE_DURATION_EXT)
	
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
		self.path = np.full((int(self.trailDuration*60/self.timescale)), ne.evaluate('sum(v)', local_dict = {'v': self.weights}), dtype=np.complex128)
		self.tail = 0
		self.generatePathColorArray()
	
	def generatePathColorArray(self):
		self.pathColorArr = np.tile([self.pathColor[0],self.pathColor[1],self.pathColor[2],0],self.path.size)
		self.pathColorArr[3::4] = np.linspace(0,1,self.path.size)
		self.pathColorArr = (GLfloat * self.pathColorArr.size)(*self.pathColorArr)
	
	def dt(self):
		return 1/60 * self.timescale
	
	def display(self):
		glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
		glClear(GL_COLOR_BUFFER_BIT)
		glBlitFramebuffer(0, 0, self.dims[0], self.dims[1], 0, 0, self.dims[0], self.dims[1], GL_COLOR_BUFFER_BIT, GL_NEAREST)
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
	
	
	def render(self, duration=CYCLE_DURATION, fps=60, fpf=1, output = 'out', show=False):
		print('Preparing render')
		self.generateTrail()
		self.vecs = np.append(0,self.weights)
		if fpf > 1:
			self.skipC = max(1,fpf-1)
			w = np.tile(np.transpose([np.append(0,self.freqs*1j)]),(1,self.skipC)) * (np.arange(1,self.skipC+1)*self.dt())[None,:]
			self.stepM = ne.evaluate('exp(w)')
			self.step = np.transpose(self.stepM[:,0])
		else:
			self.step = ne.evaluate('exp(1j*k*t)', local_dict = {'k': np.append(0,self.freqs), 't': self.dt()})
		
		#Initialize GL
		if not glfw.init():
			return
			
		if not show:
			glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
		window = glfw.create_window(self.dims[0], self.dims[1], "fourier-drawer", None, None)
		if not window:
			glfw.terminate()
			return
		glfw.make_context_current(window)
		glfw.set_window_attrib(window, glfw.RESIZABLE , glfw.FALSE)
		
		#Setup GL
		glEnableClientState(GL_VERTEX_ARRAY)
		
		self.fbo = GLuint(0)
		glGenFramebuffers(1, pointer(self.fbo))
		
		colorBuffer = GLuint(0)
		glGenRenderbuffers(1, pointer(colorBuffer))
		
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		glBindRenderbuffer(GL_RENDERBUFFER, colorBuffer)
		glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, self.dims[0], self.dims[1])
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorBuffer)

		self.renderLoop(window, duration, fps, fpf, output, show)
		glfw.terminate()
		return
	
	def renderLoop(self, window, duration, fps, fpf, output, show):
		save = True
		pT = time.time()
		s = 'XX:XX remaining'
		d100 = [0]*50
		tail = 0
		
		self.writer = skvideo.io.FFmpegWriter('{}.mp4'.format(output), inputdict={'-r': str(fps)}, outputdict={'-vcodec': 'libx264', '-vf': 'format=yuv420p'})
		while self.time < duration and not glfw.window_should_close(window):
			t = time.time()
			self.draw(save)
			if show and save:
				self.display()
				glfw.swap_buffers(window)
				glfw.poll_events()
			
			if fpf > 1:
				save = not save
			
			d100[tail] = time.time()-t
			tail = (tail+1)%50
			if time.time()-pT > 2:
				pT = time.time()
				s = (duration-self.time)/self.dt()/fpf * sum(d100)/50
				if fpf > 1:
					s*=2
				if s < 3600:
					s = '| {:02}:{:02.0f} remaining   '.format(int((s%3600)/60),s%60)
				else:
					s = '| {}:{:02}:{:02.0f} remaining  '.format(int(s/3600),int((s%3600)/60),s%60)
			printProgressBar(self.time/duration, 'Rendering', s)
		printProgressBar(1, 'Rendering', '00:00 remaining           ')
	
	def draw(self, save):
		if not save:
			pPath = np.matmul(self.vecs,self.stepM[:,:-1])
			self.vecs = ne.evaluate('a*b', local_dict = {'a': self.vecs, 'b': self.stepM[:,-1]})
			pPath = np.append(pPath, ne.evaluate('sum(v)', local_dict = {'v': self.vecs}))
			self.time += self.dt()*self.skipC
			
			if self.tail+len(pPath) >= len(self.path):
				mid = self.tail+len(pPath) - len(self.path)
				self.path[self.tail:self.tail+len(pPath)-mid] = pPath[0:len(pPath)-mid]
				self.path[:mid] = pPath[len(pPath)-mid:]
				self.tail = mid
			else:
				self.path[self.tail:self.tail+len(pPath)] = pPath
				self.tail = self.tail+len(pPath)
		else:
			self.time += self.dt()
			if len(self.step) < 300000:
				self.vecs = self.vecs * self.step
			else:
				self.vecs = ne.evaluate('a*b', local_dict = {'a': self.vecs, 'b': self.step})
			
			vecSum = np.cumsum(self.vecs)
			self.path[self.tail] = vecSum[-1]
			self.tail = (self.tail+1)%len(self.path)
			
			cutIndex = np.argmax(abs(vecSum-vecSum[-1])<1)+1
			if self.cutIndex < cutIndex:
				self.cutIndex = cutIndex
			vecSum = vecSum[0:self.cutIndex]
			
		
			glClear(GL_COLOR_BUFFER_BIT)
			glLoadIdentity()
			glScale(2/self.dims[0],2/self.dims[1],1)
			
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
				self.saveFrame()
				
	def saveFrame(self):
		data = (GLubyte * (self.dims[0] * self.dims[1] * len('RGBA')))()
		glReadPixels(0, 0, self.dims[0], self.dims[1], GL_RGBA, GL_UNSIGNED_BYTE, data)
		arr = np.flip(np.reshape(np.frombuffer(data, dtype=np.ubyte, count=self.dims[0] * self.dims[1] * len('RGBA')), (self.dims[1], self.dims[0], 4)),0)
		self.writer.writeFrame(arr)

def renderPath(path, dims, duration, timescale, trailLength, trailFade, trailColor, vectorColor, fps, fpf, output, show):
	N = len(path)
	X = np.fft.fft(path)
	freqs = np.append(np.arange(0,int(N/2)),np.arange(-int(np.ceil(N/2)),0))
	
	world = World(X/N,freqs,dims)
	world.configTime(timescale)
	world.configTrail(trailLength)
	world.setPathFade(trailFade)
	world.setPathColor(trailColor)
	world.setVectorColor(vectorColor)
	world.render(fps=fps,duration=duration,fpf=fpf,output=output,show=show)		
