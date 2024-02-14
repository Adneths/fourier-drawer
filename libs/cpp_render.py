from ctypes import WinDLL, CDLL, POINTER, byref, c_double, c_float, c_int, c_size_t, c_char_p, c_bool, Structure
import pathlib
import os
import numpy as np
import json

Vec3 = c_float * 3
class RenderParam(Structure):
	_fields_= [('x', c_float), ('y', c_float), ('width', c_float), ('height', c_float), ('zoom', c_float), ('vectorWidth', c_float), ('trailWidth', c_float), ('vectorColor', Vec3), ('trailColor', Vec3), ('fps', c_int), ('output', c_char_p), ('followTrail', c_bool), ('trailFade', c_bool)]

def hex2vec(hexCode):
	return Vec3(*[((hexCode>>16)&0xff)/255,((hexCode>>8)&0xff)/255,((hexCode)&0xff)/255]);

def printGPUInfo():
	libname = os.path.join(pathlib.Path().absolute(), 'libs\\cuda_info.dll')
	info_lib = CDLL(libname, winmode=0)
	if info_lib.printGPUInfo() != 0:
		print("Unable to get GPU info")

def renderPath(path, center, dims, view, zoom, dt, duration, start, trailLength, trailFade, followTrail, trailWidth, vectorWidth, trailColor, vectorColor, fps, fpf, output, GPU, show, flags):
	libname = os.path.join(pathlib.Path().absolute(), 'libs\\cuda_render.dll' if GPU != -1 else 'libs\\render.dll')
	for p in os.environ.get('PATH').split(';'):
		try:
			os.add_dll_directory(p)
		except Exception:
			pass;
	render_lib = CDLL(libname)

	#render(float* data, size_t size, int width, int height, float dt, float duration, float start, float trailLength, RenderParam* renders, size_t renderCount, int fpf, int gpu, bool show, bool flags)
	render_lib.render.argtypes = [POINTER(c_float), c_size_t, c_int, c_int, c_float, c_float, c_float, c_float, POINTER(RenderParam), c_size_t, c_int, c_int, c_bool, c_int]
	
	X = np.fft.fft(path)/len(path)
	data = np.empty((X.size*2), dtype=float)
	data[0::2] = np.real(X)
	data[1::2] = np.imag(X)
		
	params = []
	if output.endswith('.json'):
		with open(output, 'r') as f:
			root = json.load(f)
			for render in root['renders']:
				output_name = render.get('output', 'out.mp4')
				if not output_name.endswith('.mp4'):
					output_name += '.mp4';
				params.append(RenderParam(float(render.get('x', 0)), float(render.get('y', 0)), float(render.get('width', 800)), float(render.get('height', 800)), float(render.get('zoom', 0.9)), float(render.get('vector_width', 1)), float(render.get('trail_width', 1)), hex2vec(int(render.get('vector_color', '#ffffff')[1:], base=16)), hex2vec(int(render.get('trail_color', '#ffff00')[1:], base=16)), render.get('fps', 60), str.encode(output_name), render.get('follow_path', False), render.get('trail_fade', True)))
	if len(params) == 0:
		if not output.endswith('.mp4'):
			output += '.mp4';
		params.append(RenderParam(center[0], center[1], view[0], view[1], zoom, vectorWidth, trailWidth, hex2vec(vectorColor), hex2vec(trailColor), fps, str.encode(output), followTrail, trailFade))
	render_lib.render((c_float * len(data))(*data), len(data), dims[0], dims[1], dt, duration, start, trailLength, (RenderParam * len(params))(*params), len(params), fpf, GPU, show, flags)