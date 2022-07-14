from ctypes import *
import skvideo.io

import numpy as np
import numexpr as ne
import scipy
from PIL import Image

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders

import cupy as cp
from cuda import cudart
from .cuda_buffer import CudaOpenGLMappedArray

import time

from .util import printProgressBar

from .render import World
class CudaWorld(World):
	VERTEX_SHADER = '''
	#version 430

	layout(location = 0) in vec2 pos;
	layout(location = 2) uniform mat2 scale;
	
	flat out int ind;

	void main() {
		//gl_Position = vec4(pos.x/400,pos.y/400, 0.0, 1.0);
		gl_Position = vec4(scale * pos, 0.0, 1.0);
		ind = gl_VertexID;
	}
	'''
	
	FRAGMENT_SHADER = '''
	#version 430
	out vec4 FragColor;
	
	layout(location = 0) uniform vec3 drawColor;

	void main()
	{
		FragColor = vec4(drawColor, 1.0);
	}
	'''
	
	FRAGMENT_SHADER_FADE = '''
	#version 430
	out vec4 FragColor;
	
	flat in int ind;
	layout(location = 0) uniform vec3 drawColor;
	layout(location = 1) uniform int total;

	void main()
	{
		FragColor = vec4(drawColor, float(ind)/total);
	}
	'''
		
	def render(self, duration=World.CYCLE_DURATION, fps=60, fpf=1, output = 'out', show=False):
		print('Preparing render')
		
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
		glfw.set_window_attrib(window, glfw.RESIZABLE , glfw.FALSE);
		
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

		#Setup VBOs
		self.skipC = max(1,fpf-1)
		self.vecLength = len(self.weights)+1
		self.pathLength = int(self.trailDuration*60/self.timescale)
		
		self.shader_vec = OpenGL.GL.shaders.compileProgram(
			OpenGL.GL.shaders.compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER),
			OpenGL.GL.shaders.compileShader(self.FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
		)
		M = (c_float * 16)(2/self.dims[0],0, 0,2/self.dims[1])
		glUseProgram(self.shader_vec)
		glUniformMatrix2fv(2, 1, GL_FALSE, pointer(M))
		
		if self.pathFade:
			self.shader_path = OpenGL.GL.shaders.compileProgram(
				OpenGL.GL.shaders.compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER),
				OpenGL.GL.shaders.compileShader(self.FRAGMENT_SHADER_FADE, GL_FRAGMENT_SHADER),
			)
			glUseProgram(self.shader_path)
			glUniformMatrix2fv(2, 1, GL_FALSE, pointer(M))
			glUniform1i(1, self.pathLength)
		else:
			self.shader_path = self.shader_vec
		
		
		
		VBO_vec, VBO_path = glGenBuffers(2)
		flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
		
		glBindBuffer(GL_ARRAY_BUFFER, VBO_vec)
		glBufferData(GL_ARRAY_BUFFER, self.vecLength * np.complex128().nbytes, None, GL_DYNAMIC_DRAW)
		self.buffer_vec = CudaOpenGLMappedArray(np.complex128, (self.vecLength), VBO_vec, flags)
		
		glBindBuffer(GL_ARRAY_BUFFER, VBO_path)
		glBufferData(GL_ARRAY_BUFFER, self.pathLength * np.complex128().nbytes, None, GL_DYNAMIC_DRAW)
		self.buffer_path = CudaOpenGLMappedArray(np.complex128, (self.pathLength), VBO_path, flags)
		with self.buffer_path as V:
			V[:] = cp.full((self.pathLength), np.sum(self.weights * ne.evaluate('exp(1j*k*t)', local_dict = {'k': self.freqs, 't': self.time})), dtype=np.complex128)

		self.vecs = cp.append(0,cp.asarray(self.weights))
		self.stepM = cp.exp(cp.tile(cp.reshape(cp.append(0,cp.asarray(self.freqs)*1j),(self.vecLength,1)),(1,self.skipC)) * (cp.arange(1,self.skipC+1)*self.dt())[None,:])
		
		self.cutIndex = 0
		save = True
		pT = time.time()
		s = 'XX:XX remaining'
		d100 = [0]*50
		tail = 0
		
		self.writer = skvideo.io.FFmpegWriter('{}.mp4'.format(output), inputdict={'-r': str(fps)}, outputdict={'-vcodec': 'libx264', '-vf': 'format=yuv420p'})
		while self.time < duration and not glfw.window_should_close(window):
			t = time.time()
			self.draw(save)
			if show:
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
			printProgressBar(self.time/duration, 'C:Rendering', s)
		printProgressBar(1, 'C:Rendering', '00:00 remaining           ')
		self.buffer_vec.unregister()
		self.buffer_path.unregister()
		glfw.terminate()
		return
	
	def draw(self, save):
		if not save:
			self.time += self.dt()*self.skipC
			with self.buffer_vec as V:
				pPath = cp.matmul(self.vecs,self.stepM[:,:-1])
				self.vecs = self.vecs * self.stepM[:,-1]
				with self.buffer_path as P:
					P[:] = cp.roll(P,-self.skipC)
					P[-self.skipC:-1] = pPath
					P[-1] = cp.sum(self.vecs)
		else:
			self.time += self.dt()
			with self.buffer_vec as V:
				a = self.stepM[:,0]
				self.vecs = self.vecs*self.stepM[:,0]
				vecSum = cp.cumsum(self.vecs)
				cutIndex = cp.argmax(abs(vecSum-vecSum[-1])<1)
				if self.cutIndex < cutIndex:
					self.cutIndex = cutIndex
				V[0:self.cutIndex] = vecSum[0:self.cutIndex]
				with self.buffer_path as P:
					P[:] = cp.roll(P,-1)
					P[-1] = vecSum[-1]
		
			glClear(GL_COLOR_BUFFER_BIT)
			glLoadIdentity()
			
			
			glLineWidth(1)
			glBindBuffer(GL_ARRAY_BUFFER, self.buffer_vec.gl_buffer)
			glVertexAttribPointer(0, 2, GL_DOUBLE, GL_FALSE, 2* sizeof(c_double), c_void_p(0))
			glEnableVertexAttribArray(0)
			
			glUseProgram(self.shader_vec)
			glUniform3f(0, self.vecColor[0],self.vecColor[1],self.vecColor[2])
			glDrawArrays(GL_LINE_STRIP, 0, self.cutIndex);
			
			
			glLineWidth(1.5)
			glEnable(GL_BLEND)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			
			glBindBuffer(GL_ARRAY_BUFFER, self.buffer_path.gl_buffer)
			glVertexAttribPointer(0, 2, GL_DOUBLE, GL_FALSE, 2* sizeof(c_double), c_void_p(0))
			glEnableVertexAttribArray(0)
			
			glUseProgram(self.shader_path)
			glUniform3f(0, self.pathColor[0],self.pathColor[1],self.pathColor[2])
			glDrawArrays(GL_LINE_STRIP, 0, self.pathLength);
			
			glDisable(GL_BLEND)
			
			if self.writer != None:
				self.saveFrame()


def renderPath(path, dims, duration, timescale, trailLength, trailFade, trailColor, vectorColor, fps, fpf, output, show, gpu=None):
	if gpu != None:
		cp.cuda.Device(gpu).use()
	N = len(path)
	X = np.fft.fft(path)
	freqs = np.append(np.arange(0,int(N/2)),np.arange(-int(np.ceil(N/2)),0))
	
	world = CudaWorld(X/N,freqs,dims)
	world.configTime(timescale)
	world.configTrail(trailLength)
	world.setPathFade(trailFade)
	world.setPathColor(trailColor)
	world.setVectorColor(vectorColor)
	world.render(fps=fps,duration=duration,fpf=fpf,output=output,show=show)		

