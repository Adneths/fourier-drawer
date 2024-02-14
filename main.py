import argparse
from libs.path import *

from libs.cpp_render import renderPath, printGPUInfo

def strMath(s, var = {}):
	ops = [{'*': lambda a,b: a*b, '/': lambda a,b: a/b}, {'+': lambda a,b: a+b, '-': lambda a,b: a-b}]
	s = ''.join(s.split())
	m = re.findall(r'(\+|-|\*|\/|[0-9]+(\.[0-9]+)?|[Pp][Ii]|${[a-zA-z]+})', s)
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
infoMap = {'d': 1, 'p': 2, 'r': 4, 'g': 8}
def infoBits(s):
	flags = 0
	for c in s:
		flags |= infoMap.get(c, 0)
	return int(flags);

parser = argparse.ArgumentParser(description='Converts input file into a fourier series')

#Input Parameter
group_inputs = parser.add_argument_group("Input Parameter")
group_inputs.add_argument('-i', '--input', type=str, required=True, help='the input file')
group = group_inputs.add_mutually_exclusive_group(required=True)
group.add_argument('-s', '--svg', action='store_true', help="marks the input file as a svg")
group.add_argument('-b', '--bitmap', action='store_true', help="marks the input file as a bitmap type (bmp, png, jpg, etc.)")
group.add_argument('-v', '--video', action='store_true', help="marks the input file as a video type (mp4, avi, mov, etc.)")
group.add_argument('-p', '--path', action='store_true', help="marks the input file as a .npy (numpy array) file)")
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
group_render.add_argument('-tl', '--trail-length', type=str, default='2.1*pi', help='the duration of video time to keep the trail visible (2pi video time seconds is 1 cycle). Accepts math expressions including (+,-,*,/,pi,${frames})')
group = group_render.add_mutually_exclusive_group()
group.add_argument('-tf', '--trail-fade', action='store_true', help='whether the tail should fade with time')
group.add_argument('-ntf', '--no-trail-fade', action='store_false', help='whether the tail should fade with time')
group_render.add_argument('-tc', '--trail-color', type=str, default='#ffff00', help='\'#xxxxxx\' color of the trail as a hexcode')
group_render.add_argument('-vc', '--vector-color', type=str, default='#ffffff', help='\'#xxxxxx\' color of the vectors as a hexcode')
group_render.add_argument('-tw', '--trail-width', type=float, default=1, help='width of the trail')
group_render.add_argument('-vw', '--vector-width', type=float, default=1, help='width of the vectors')
group_render.add_argument('-fpf', '--frames-per-frame', type=str, default='1', help='one video frame is saved every this many dts. There are 2*pi*60/{timescale} frames in a render. Accepts math expressions including (+,-,*,/,pi,${frames}) casted to int')
group_render.add_argument('--center', type=str, default='0x0', help='\'[x]x[y]\' offset from the center')
group_render.add_argument('-view', '--viewport', type=str, default=None, help='\'[width]x[height]\' dimensions of the output video (defaults to image/video dimensions, or 800x800 for svg)')
group_render.add_argument('-z', '--zoom', type=float, default=0.9, help='percentage (as a float) of border between the path and screen')
group_render.add_argument('-ft', '--follow-trail', action='store_true', help="centers video on the head of vectors (includs offset)")

group_render.add_argument('-g', '--gpu', type=str, nargs='?', const='0', help='use Cuda to accelerate rendering process (use a number to specify a GPU or ? to list avaliable GPUs)')

#Debug Parameter
group_debug = parser.add_argument_group("Debug Parameter")
group_debug.add_argument('--info', type=str, default='', help='d for Debug, p for Path, r for Render, g for GPU')
group_debug.add_argument('--profile', action='store_true', help='profiles timing information')


#parser.add_argument('--show', action='store_true', help='Display the sketch during rendering')
#parser.add_argument('-m-lim', '--memory-limit', type=str, default='2G', help='(Approximate) Sets the maximum amount of memory the program should use during rendering. If it is insufficient the program will request for more. Accepts a number followed by a unit (K,M,G)')

args = parser.parse_args()

if args.gpu != None:
	if args.gpu[0] == '?':
		printGPUInfo()
		exit(0)
	else:
		GPU = int(args.gpu)
else:
	GPU = -1

vColor = int(args.vector_color[1:], base=16)
tColor = int(args.trail_color[1:], base=16)

if args.dimension != None:
	s = ''.join(args.dimension.split()).split('x')
	dims = (int(s[0])//10*10,int(s[1])//10*10)
else:
	dims = None
if args.viewport != None:
	s = ''.join(args.viewport.split()).split('x')
	view = (int(s[0])//10*10,int(s[1])//10*10)
else:
	view = None

if args.center != None:
	s = ''.join(args.center.split()).split('x')
	center = (float(s[0]),float(s[1]))
else:
	center = (0,0)

frames = 1
if args.svg:
	if dims == None:
		dims = (800,800)
	print('Tracing image')
	path = boundPath(centerPath(svgToPath(args.input, abs(args.density), args.points)), (dims[0]/min(dims),dims[1]/min(dims)))
elif args.bitmap:
	if dims == None:
		size = Image.open(args.input).size
		dims = (int(size[0]/10)*10,int(size[1]/10)*10)
	print('Tracing image')
	path = boundPath(centerPath(imageFileToPath(args.input, abs(args.density), args.points)), (dims[0]/min(dims),dims[1]/min(dims)))
elif args.video:
	print('Tracing video')
	path, dims, frames = videoToPath(args.input, abs(args.density), args.points, dims, args.border)
	path = boundPath(centerPath(path), (dims[0]/min(dims),dims[1]/min(dims)))
elif args.path:
	print('Reading path' ,end='')
	data = np.load(args.input)
	path = data[2:]
	dims = (int(np.real(data[1])),int(np.imag(data[1])))
	frames = int(np.real(data[0]))

if view == None:
	view = dims
	
	
flags = infoBits(args.info)
if args.profile:
	flags |= 16

print()
if (flags & 2) != 0:
	print('Number of points:', len(path))
	print('Output dimensions:', dims)
	if frames != 1:
		print('Input frames:', frames)

if args.save_path != None:
	data = np.append(np.asarray([frames, dims[0] + dims[1]*1j], dtype=np.complex128), path)
	np.save(args.save_path, data)
	
var = {'frames': frames}
timescale = strMath(args.timescale, var)
duration = strMath(args.duration, var)
trailLength = strMath(args.trail_length, var)
fpf = int(strMath(args.frames_per_frame, var))
start = strMath(args.start, var)
#memLim = strToMemory(args.memory_limit)


print('Loading Libraries')
renderPath(path, center, dims, view, args.zoom, timescale/60, duration, start, trailLength, args.trail_fade or args.no_trail_fade, args.follow_trail, args.trail_width, args.vector_width, tColor, vColor, args.fps, fpf, args.output, GPU, False, flags)