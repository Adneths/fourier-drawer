from ctypes import CDLL, POINTER, byref, c_double, c_float, c_int, c_size_t, c_char_p, c_bool, Structure
import pathlib
import os
import numpy as np

import pdb

Vec3 = c_float * 3
class RenderParam(Structure):
	_fields_= [('x', c_float), ('y', c_float), ('width', c_float), ('height', c_float), ('vectorWidth', c_float), ('trailWidth', c_float), ('vectorColor', Vec3), ('trailColor', Vec3), ('fps', c_int), ('output', c_char_p), ('followPath', c_bool), ('trailFade', c_bool)]

def hex2vec(hexCode):
	return Vec3(*[((hexCode>>16)&0xff)/255,((hexCode>>8)&0xff)/255,((hexCode)&0xff)/255]);

def renderPath(path, dims, dt, duration, start, trailLength, trailFade, trailColor, vectorColor, fps, fpf, output, cuda, show, debug):
	if cuda:
		libname = os.path.join(pathlib.Path().absolute(), 'libs\\cuda_render.dll')
	else:
		libname = os.path.join(pathlib.Path().absolute(), 'libs\\render.dll')
	render_lib = CDLL(libname, winmode=0)

	#render(float* data, size_t size, float dt, float duration, float start, float trailLength, RenderParam* renders, size_t renderCount, int fpf, bool show, bool debug)
	render_lib.render.argtypes = [POINTER(c_float), c_size_t, c_float, c_float, c_float, c_float, POINTER(RenderParam), c_size_t, c_int, c_bool, c_bool]
	
	X = np.fft.fft(path)/len(path)
	data = np.empty((X.size*2), dtype=float)
	data[0::2] = np.real(X)
	data[1::2] = np.imag(X)
	if not output.endswith('.mp4'):
		output += '.mp4';
	param = RenderParam(0, 0, dims[0], dims[1], 1, 1, hex2vec(vectorColor), hex2vec(trailColor), fps, str.encode(output), False, trailFade)
	render_lib.render((c_float * len(data))(*data), len(data), dt, duration, start, trailLength, param, 1, fpf, show, debug)