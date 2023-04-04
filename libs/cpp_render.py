from ctypes import CDLL, POINTER, byref, c_double, c_float, c_int, c_size_t, c_char_p
import pathlib
import os
import numpy as np

Vec3 = c_float * 3

def hex2vec(hexCode):
	return Vec3(*[((hexCode>>16)&0xff)/255,((hexCode>>8)&0xff)/255,((hexCode)&0xff)/255]);

def renderPath(path, dims, dt, duration, start, trailLength, trailColor, vectorColor, fps, fpf, output):
	libname = os.path.join(pathlib.Path().absolute(), 'libs/render.dll')
	render_lib = CDLL(libname, winmode=0)

	#render(float* data, size_t size, int width, int height, float dt, float duration, float start, float trailLength, glm::vec3 trailColor, glm::vec3 vectorColor, int fps, int fpf, const char* output)
	render_lib.render.argtypes = [POINTER(c_float), c_size_t, c_int, c_int, c_float, c_float, c_float, c_float, Vec3, Vec3, c_int, c_int, c_char_p]
	
	X = np.fft.fft(path)/len(path)
	data = np.empty((X.size*2), dtype=float)
	data[0::2] = np.real(X)
	data[1::2] = np.imag(X)
	if not output.endswith('.mp4'):
		output += '.mp4';
	print(render_lib.render((c_float * len(data))(*data), len(data), dims[0], dims[1], dt, duration, start, trailLength, hex2vec(trailColor), hex2vec(vectorColor), fps, fpf, str.encode(output)))