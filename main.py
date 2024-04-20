import argparse
import re
from libs.path import *

from libs.cpp_render import *

def strMath(s, var = {}):
	ops = [{'*': lambda a,b: a*b, '/': lambda a,b: a/b}, {'+': lambda a,b: a+b, '-': lambda a,b: a-b}]
	s = ''.join(s.split())
	m = re.findall(r'(\+|-|\*|\/|[0-9]+(\.[0-9]+)?|[Pp][Ii]|\$\{[a-zA-z]+\})', s)
	exp = []
	for sym in m:
		if sym[0].lower() == 'pi':
			exp.append(np.pi)
		elif re.match('\\+|-|\\*|\\/', sym[0]):
			exp.append(sym[0])
		elif sym[0][0:2] == '${' and sym[0][-1] == '}':
			exp.append(var[sym[0][2:-1]])
		else:
			exp.append(float(sym[0]))
	for n in range(2):
		i = 1
		while i < len(exp):
			if exp[i] in ops[n]:
				exp[i-1] = ops[n][exp[i]](exp[i-1],exp[i+1])
				exp.pop(i)
				exp.pop(i)
			else:
				i+=1
	return exp[0]

def strToMemory(s):
	s = ''.join(s.split())
	factor = {'k': 1024, 'm': 1024**2, 'g': 1024**3}
	return float(s[:-1]) * factor[s[-1].lower()]
	
'''
Debug, Path, Render, GPU
'''
infoMap = {'d': 1, 'p': 2, 'r': 4, 'g': 8, 'w': 32}
def infoBits(s):
	flags = 0
	for c in s:
		flags |= infoMap.get(c, 0)
	return int(flags);

parser = argparse.ArgumentParser(description='Converts input file into a fourier series')

#Input Parameter
group_inputs = parser.add_argument_group("Input Parameter")
group_inputs.add_argument('-i', '--input', type=str, required=True, help='the input file')
group = group_inputs.add_mutually_exclusive_group(required=False)
group.add_argument('-s', '--svg', action='store_true', default=False, help="marks the input file as a svg")
group.add_argument('-b', '--bitmap', action='store_true', default=False, help="marks the input file as a bitmap type (bmp, png, jpg, etc.)")
group.add_argument('-v', '--video', action='store_true', default=False, help="marks the input file as a video type (mp4, avi, mov, etc.)")
group.add_argument('-p', '--path', action='store_true', default=False, help="marks the input file as a .npy (numpy array file)")
group.add_argument('-j', '--json', action='store_true', default=False, help="marks the input file as a .json (options file)")
group = group_inputs.add_mutually_exclusive_group()
group.add_argument('--density', type=float, default=2, help='how densely packed are samples of a path')
group.add_argument('--points', type=int, default=-1, help='how many point in an image or frame')

#Output Parameter
group_outputs = parser.add_argument_group("Output Parameter")
group_outputs.add_argument('-o', '--output', type=str, default='out', help='the output file name')
group_outputs.add_argument('-dim', '--dimension', type=str, default=None, help='\'[width]x[height]\' dimensions of the input (defaults to image/video dimensions, or 800x800 for svg)')
group_outputs.add_argument('-fps', type=int, default=60, help='fps of the output video')
group_outputs.add_argument('--save-path', type=str, default=None, help='saves the path in a file to save recomputation')

#Render Parameter
group_render = parser.add_argument_group("Render Parameter")
group_render.add_argument('-t', '--timescale', type=str, default='1', help='how many seconds video time is 1 second real time (2pi video time seconds is 1 cycle). Accepts math expressions including (+,-,*,/,pi,${frames})')
group_render.add_argument('-d', '--duration', type=str, default='2*pi', help='the duration of video time to write to file (2pi video time seconds is 1 cycle). Accepts math expressions including (+,-,*,/,pi,${frames})')
group_render.add_argument('-ss', '--start', type=str, default='0', help='the time after which writing to file begins (2pi video time seconds is 1 cycle). Accepts math expressions including (+,-,*,/,pi,${frames})')
group_render.add_argument('-pl', '--path-length', type=str, default='2.1*pi', help='the duration of video time to keep the path visible (2pi video time seconds is 1 cycle). Accepts math expressions including (+,-,*,/,pi,${frames})')
group = group_render.add_mutually_exclusive_group()
group.add_argument('-pf', '--path-fade', action='store_true', help='whether the tail should fade with time')
group.add_argument('-npf', '--no-path-fade', action='store_false', help='whether the tail should fade with time')
group_render.add_argument('-pc', '--path-color', type=str, default='#ffff00', help='\'#xxxxxx\' color of the path as a hexcode')
group_render.add_argument('-vc', '--vector-color', type=str, default='#ffffff', help='\'#xxxxxx\' color of the vectors as a hexcode')
group_render.add_argument('-pw', '--path-width', type=float, default=1, help='width of the path')
group_render.add_argument('-vw', '--vector-width', type=float, default=1, help='width of the vectors')
group_render.add_argument('-spf', '--steps-per-frame', type=str, default='1', help='one video frame is saved every this many timesteps. There are 2*pi*60/{timescale} timesteps in a render. Accepts math expressions including (+,-,*,/,pi,${frames}) casted to int')
group_render.add_argument('--center', type=str, default='0x0', help='\'[x]x[y]\' offset from the center')
group_render.add_argument('--screen', type=str, default=None, help='\'[width]x[height]\' dimensions of the output video (defaults to image/video dimensions, or 800x800 for svg)')
group_render.add_argument('-z', '--zoom', type=float, default=0.9, help='percentage (as a float) of border between the path and screen')
group_render.add_argument('-fp', '--follow-path', action='store_true', help="centers video on the head of vectors (includs offset)")

group_render.add_argument('-g', '--gpu', type=str, nargs='?', const='0', help='use Cuda to accelerate rendering process (use a number to specify a GPU or ? to list avaliable GPUs)')

#Debug Parameter
group_debug = parser.add_argument_group("Debug Parameter")
group_debug.add_argument('--info', type=str, default='', help='d for Debug, p for Path, r for Render, g for GPU, w for warnings')
group_debug.add_argument('--profile', action='store_true', help='profiles timing information')


#parser.add_argument('--show', action='store_true', help='Display the sketch during rendering')
#parser.add_argument('-m-lim', '--memory-limit', type=str, default='2G', help='(Approximate) Sets the maximum amount of memory the program should use during rendering. If it is insufficient the program will request for more. Accepts a number followed by a unit (K,M,G)')

print('Loading Parameters')
args = parser.parse_args()

GPU = args.gpu
DENSITY = args.density
POINTS = args.points
VECTOR_COLOR = args.vector_color
PATH_COLOR = args.path_color
DIMENSION = args.dimension
SCREEN = args.screen
CENTER = args.center
SVG = args.svg
BITMAP = args.bitmap
VIDEO = args.video
PATH = args.path
JSON = args.json
SAVE_PATH = args.save_path
ZOOM = args.zoom

TIMESCALE = args.timescale
DURATION = args.duration
PATH_LENGTH = args.path_length
STEPS_PER_FRAME = args.steps_per_frame
START = args.start

INPUT = args.input
OUTPUT = args.output


vColor = int(VECTOR_COLOR[1:], base=16)
pColor = int(PATH_COLOR[1:], base=16)

if DIMENSION != None:
	s = ''.join(DIMENSION.split()).split('x')
	dims = (int(s[0])//10*10,int(s[1])//10*10)
else:
	dims = None
if SCREEN != None:
	s = ''.join(SCREEN.split()).split('x')
	screen = (int(s[0])//10*10,int(s[1])//10*10)
else:
	screen = None

if CENTER != None:
	s = ''.join(CENTER.split()).split('x')
	center = (float(s[0]),float(s[1]))
else:
	center = (0,0)

frames = 1
params = []
none_set = not (SVG or BITMAP or VIDEO or PATH or JSON)

if args.json or re.search('\\.(json)$', INPUT)!=None:
	with open(INPUT, 'r') as f:
		root = json.load(f)
		SAVE_PATH = root.get('save_path', SAVE_PATH)
		if root.get('input', None) != None:
			types = {
				's': (True, False, False, False),
				'b': (False, True, False, False),
				'v': (False, False, True, False),
				'p': (False, False, False, True),
				'svg': (True, False, False, False),
				'bitmap': (False, True, False, False),
				'video': (False, False, True, False),
				'path': (False, False, False, True)
			}
			SVG, BITMAP, VIDEO, PATH = types[root['input'].get('type', None)] if root['input'].get('type', None) != None else (SVG, BITMAP, VIDEO, PATH)
			INPUT = root['input'].get('input_name', INPUT)
			DENSITY = root['input'].get('density', DENSITY)
			POINTS = root['input'].get('points', POINTS)
			DIMENSION = root['input'].get('dimension', DIMENSION)
		
		if root.get('render', None) != None:
			TIMESCALE = root['render'].get('timescale', TIMESCALE)
			DURATION = root['render'].get('duration', DURATION)
			START = root['render'].get('start', START)
			STEPS_PER_FRAME = root['render'].get('steps_per_frame', STEPS_PER_FRAME)
			PATH_LENGTH = root['render'].get('path_length', PATH_LENGTH)
			GPU = root['render'].get('gpu', GPU)
			if root['render'].get('outputs', None) != None:
				for output in root['render']['outputs']:
					output_name = output.get('output', 'out.mp4')
					if not output_name.endswith('.mp4'):
						output_name += '.mp4';
					param = RenderParam(str.encode(output_name), int(output.get('width', 800)), int(output.get('height', 800)), int(output.get('fps', 60)))
					for i, view in enumerate(output['views']):
						param.views[i] = View(True, view.get('no_background', False), hex2vec(int(view.get('background_color', 0x000000)[1:], base=16)), hex2vec(int(view.get('border_color', 0x000000)[1:], base=16)), float(view.get('border_width', 0)), view.get('border_on_other_views', False), int(view.get('screen_x', 0)), int(view.get('screen_y', 0)), int(view.get('screen_width', 800)), int(view.get('screen_height', 800)), float(view.get('center_x', 0)), float(view.get('center_y', 0)), float(view.get('zoom', 1)), float(view.get('vector_width', 1)), float(view.get('path_width', 1)), hex2vec(int(view.get('vector_color', 0xffffff)[1:], base=16)), hex2vec(int(view.get('path_color', 0xffff00)[1:], base=16)), bool(view.get('follow_path', False)), bool(view.get('path_fade', True)))
					params.append(param)

if GPU != None:
	if GPU[0] == '?':
		printGPUInfo()
		exit(0)
	else:
		GPU = int(GPU)
else:
	GPU = -1

if SVG or none_set and re.search('\\.(svg)$', INPUT)!=None:
	if dims == None:
		dims = (800,800)
	print('Tracing image')
	path = boundPath(centerPath(svgToPath(INPUT, abs(DENSITY), POINTS)), (dims[0]/min(dims),dims[1]/min(dims)))
elif BITMAP or none_set and re.search('\\.(bmp|png|jpg|jpeg)$', INPUT)!=None:
	if dims == None:
		size = Image.open(INPUT).size
		dims = (int(size[0]/10)*10,int(size[1]/10)*10)
	print('Tracing image')
	path = boundPath(centerPath(imageFileToPath(INPUT, abs(DENSITY), POINTS)), (dims[0]/min(dims),dims[1]/min(dims)))
elif VIDEO or none_set and re.search('\\.(mp4|avi|mov)$', INPUT)!=None:
	print('Tracing video')
	path, dims, frames = videoToPath(INPUT, abs(DENSITY), POINTS, dims)
	path = boundPath(centerPath(path), (dims[0]/min(dims),dims[1]/min(dims)))
elif PATH or none_set and re.search('\\.(npy)$', INPUT)!=None:
	print('Reading path' ,end='')
	data = np.load(INPUT)
	path = data[2:]
	dims = (int(np.real(data[1])),int(np.imag(data[1])))
	frames = int(np.real(data[0]))
else:
	print('Error: Unable to deduce input file type')
	exit()
if screen == None:
	screen = dims
	
if len(params) == 0:
	if not OUTPUT.endswith('.mp4'):
		OUTPUT += '.mp4';
	param = RenderParam(str.encode(OUTPUT), screen[0], screen[1], args.fps)
	param.views[0] = View(True, False, hex2vec(0x000000), hex2vec(0x000000), 0, False, 0, 0, screen[0], screen[1], center[0], center[1], ZOOM, args.vector_width, args.path_width, hex2vec(vColor), hex2vec(pColor), args.follow_path, args.path_fade or args.no_path_fade)
	params.append(param)
	
flags = infoBits(args.info)
if args.profile:
	flags |= 16

print()
if (flags & 2) != 0:
	print('Number of points:', len(path))
	print('Output dimensions:', dims)
	if frames != 1:
		print('Input frames:', frames)

if SAVE_PATH != None and SAVE_PATH != '':
	data = np.append(np.asarray([frames, dims[0] + dims[1]*1j], dtype=np.complex128), path)
	np.save(SAVE_PATH, data)


var = {'frames': frames}
timescale = strMath(TIMESCALE, var)
duration = strMath(DURATION, var)
pathLength = strMath(PATH_LENGTH, var)
spf = int(strMath(STEPS_PER_FRAME, var))
start = strMath(START, var)
#memLim = strToMemory(args.memory_limit)

print('Loading Libraries')
renderPath(params, path, dims, timescale/60, duration, start, pathLength, spf, GPU, False, flags)