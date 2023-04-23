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

from .render import World, computeMatSize

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
		
	def render(self, duration=World.CYCLE_DURATION, fps=60, fpf=1, output = 'out', show=False, matSize=1):
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
		self.stepM = cp.exp(cp.tile(cp.reshape(cp.append(0,cp.asarray(self.freqs)*1j),(self.vecLength,1)),(1,min(fpf,matSize))) * (cp.arange(1,min(fpf,matSize)+1)*self.dt())[None,:])
		
		glClearColor(0,0,0,1)
		
		self.renderLoop(window, duration, fps, fpf, output, show, matSize)
		self.buffer_vec.unregister()
		self.buffer_path.unregister()
		glfw.terminate()
		return

	def computeFrame(self, steps, render):
		self.time += self.dt()*steps
		with self.buffer_vec as V:
			pPath = None if steps==1 else cp.matmul(self.vecs,self.stepM[:,:steps-1])
			np.multiply(self.vecs, self.stepM[:,steps-1], out=self.vecs)#self.vecs = self.vecs * self.stepM[:,steps-1]
			if render:
				cp.cumsum(self.vecs, out=V)
				cutIndex = cp.argmax(abs(V-V[-1])<1)+1
				if self.cutIndex < cutIndex:
					self.cutIndex = cutIndex
				
				last = V[-1]
			else:
				last = cp.sum(self.vecs)
			with self.buffer_path as P:
				P[:] = cp.roll(P,-steps)
				if not pPath is None:
					P[-steps:-1] = pPath
				P[-1] = last
		return 0 if render else None
		
	def renderFrame(self, vecSum):
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

def renderPath(path, dims, duration, timescale, trailLength, trailFade, trailColor, vectorColor, fps, fpf, output, show, memLim, start, gpu=None):
	if gpu != None:
		cp.cuda.Device(gpu).use()
	N = len(path)
	X = np.fft.fft(path)
	freqs = np.append(np.arange(0,int(N/2)),np.arange(-int(np.ceil(N/2)),0))
	
	matSize = computeMatSize(memLim - 2*X.nbytes - freqs.nbytes - int(trailLength*60/timescale)*np.dtype(np.complex128).itemsize, X.nbytes, fpf)
	
	world = CudaWorld(X/N,freqs,dims)
	world.configTime(timescale)
	world.configTrail(trailLength)
	world.setPathFade(trailFade)
	world.setPathColor(trailColor)
	world.setVectorColor(vectorColor)
	world.setStartTime(start)
	world.render(fps=fps,duration=duration,fpf=fpf,output=output,show=show,matSize=matSize)
	