from ctypes import WinDLL, CDLL, POINTER, byref, c_double, c_float, c_int, c_size_t, c_char_p, c_bool, Structure
import pathlib
import os
import numpy as np
import json

# https://stackoverflow.com/a/25892189
class BaseStructure(Structure):
    def __init__(self, **kwargs):
        values = type(self)._defaults_.copy()
        values.update(kwargs)
        super().__init__(**values)
    def __init__(self, *args):
        super().__init__(*args)

Vec3 = c_float * 3
class View(BaseStructure):
	_fields_ = [
		('valid', c_bool),
		('no_background', c_bool),
		('background_color', Vec3),
		('border_color', Vec3),
		('border_width', c_float),
		('border_on_other_views', c_bool),
		('screen_x', c_int),
		('screen_y', c_int),
		('screen_width', c_int),
		('screen_height', c_int),
		('center_x', c_double),
		('center_y', c_double),
		('zoom', c_double),
		('vector_width', c_float),
		('path_width', c_float),
		('vector_color', Vec3),
		('path_color', Vec3),
		('follow_path', c_bool),
		('path_fade', c_bool)
	]
	_defaults_ = {'valid': False}
class RenderParam(BaseStructure):
	_fields_= [
		('output', c_char_p),
		('width', c_int),
		('height', c_int),
		('fps', c_int),
		('views', View*8)
	]

def hex2vec(hexCode):
	return Vec3(*[((hexCode>>16)&0xff)/255,((hexCode>>8)&0xff)/255,((hexCode)&0xff)/255]);

def printGPUInfo():
	libname = os.path.join(pathlib.Path().absolute(), 'libs\\cuda_info.dll')
	info_lib = CDLL(libname, winmode=0)
	if info_lib.printGPUInfo() != 0:
		print("Unable to get GPU info")

def renderPath(params, path, dims, dt, duration, start, pathLength, spf, GPU, precision, show, flags):
	libname = os.path.join(pathlib.Path().absolute(), 'libs\\cuda_render.dll' if GPU != -1 else 'libs\\render.dll')
	for p in os.environ.get('PATH').split(';'):
		try:
			os.add_dll_directory(p)
		except Exception:
			pass;
	render_lib = CDLL(libname)

	if precision == 'single':

		#render(float* data, size_t size, int width, int height, float dt, float duration, float start, float pathLength, RenderParam* renders, size_t renderCount, int spf, int gpu, bool show, int flags)
		render_lib.render.argtypes = [POINTER(c_float), c_size_t, c_int, c_int, c_float, c_float, c_float, c_float, POINTER(RenderParam), c_size_t, c_int, c_int, c_bool, c_int]
		
		X = np.fft.fft(path)/len(path)
		data = np.empty((X.size*2), dtype=np.float32)
		data[0::2] = np.real(X)
		data[1::2] = np.imag(X)
		
		render_lib.render((c_float * len(data))(*data), len(data), dims[0], dims[1], dt, duration, start, pathLength, (RenderParam * len(params))(*params), len(params), spf, GPU, show, flags)
	
	elif precision == 'double':

		#render_double(double* data, size_t size, int width, int height, double dt, double duration, double start, double pathLength, RenderParam* renders, size_t renderCount, int spf, int gpu, bool show, int flags)
		render_lib.render_double.argtypes = [POINTER(c_double), c_size_t, c_int, c_int, c_double, c_double, c_double, c_double, POINTER(RenderParam), c_size_t, c_int, c_int, c_bool, c_int]
		
		X = np.fft.fft(path)/len(path)
		data = np.empty((X.size*2), dtype=np.float64)
		data[0::2] = np.real(X)
		data[1::2] = np.imag(X)
		
		render_lib.render_double((c_double * len(data))(*data), len(data), dims[0], dims[1], dt, duration, start, pathLength, (RenderParam * len(params))(*params), len(params), spf, GPU, show, flags)
		