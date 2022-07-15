#Adapted from https://gist.github.com/keckj/e37d312128eac8c5fca790ce1e7fc437

from cuda import cudart

import numpy as np
import cupy as cp

def format_cudart_err(err):
	return (
		f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
		f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
	)


def check_cudart_err(args):
	if isinstance(args, tuple):
		assert len(args) >= 1
		err = args[0]
		if len(args) == 1:
			ret = None
		elif len(args) == 2:
			ret = args[1]
		else:
			ret = args[1:]
	else:
		err = args
		ret = None

	assert isinstance(err, cudart.cudaError_t), type(err)
	if err != cudart.cudaError_t.cudaSuccess:
		raise RuntimeError(format_cudart_err(err))

	return ret

class CudaOpenGLMappedBuffer:
	def __init__(self, gl_buffer, flags=0):
		self._gl_buffer = int(gl_buffer)
		self._flags = int(flags)

		self._graphics_ressource = None
		self._cuda_buffer = None

		self.register()

	@property
	def gl_buffer(self):
		return self._gl_buffer

	@property
	def cuda_buffer(self):
		assert self.mapped
		return self._cuda_buffer

	@property
	def graphics_ressource(self):
		assert self.registered
		return self._graphics_ressource

	@property
	def registered(self):
		return self._graphics_ressource is not None

	@property
	def mapped(self):
		return self._cuda_buffer is not None

	def __enter__(self):
		return self.map()

	def __exit__(self, exc_type, exc_value, trace):
		self.unmap()
		return False

	def __del__(self):
		self.unregister()

	def register(self):
		if self.registered:
			return self._graphics_ressource
		self._graphics_ressource = check_cudart_err(
			cudart.cudaGraphicsGLRegisterBuffer(self._gl_buffer, self._flags)
		)
		return self._graphics_ressource

	def unregister(self):
		if not self.registered:
			return self
		self.unmap()
		self._graphics_ressource = check_cudart_err(
			cudart.cudaGraphicsUnregisterResource(self._graphics_ressource)
		)
		return self

	def map(self, stream=None):
		if not self.registered:
			raise RuntimeError("Cannot map an unregistered buffer.")
		if self.mapped:
			return self._cuda_buffer

		check_cudart_err(
			cudart.cudaGraphicsMapResources(1, self._graphics_ressource, stream)
		)

		ptr, size = check_cudart_err(
			cudart.cudaGraphicsResourceGetMappedPointer(self._graphics_ressource)
		)

		self._cuda_buffer = cp.cuda.MemoryPointer(
			cp.cuda.UnownedMemory(ptr, size, self), 0
		)

		return self._cuda_buffer

	def unmap(self, stream=None):
		if not self.registered:
			raise RuntimeError("Cannot unmap an unregistered buffer.")
		if not self.mapped:
			return self

		self._cuda_buffer = check_cudart_err(
			cudart.cudaGraphicsUnmapResources(1, self._graphics_ressource, stream)
		)

		return self


class CudaOpenGLMappedArray(CudaOpenGLMappedBuffer):
	def __init__(self, dtype, shape, gl_buffer, flags=0, strides=None, order='C'):
		super().__init__(gl_buffer, flags)
		self._dtype = dtype
		self._shape = shape
		self._strides = strides
		self._order = order

	@property
	def cuda_array(self):
		assert self.mapped
		return cp.ndarray(
			shape=self._shape,
			dtype=self._dtype,
			strides=self._strides,
			order=self._order,
			memptr=self._cuda_buffer,
		)

	def map(self, *args, **kwargs):
		super().map(*args, **kwargs)
		return self.cuda_array