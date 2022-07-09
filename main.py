import argparse
from fourier import *

def strMath(s):
	ops = [{'*': lambda a,b: a*b, '/': lambda a,b: a/b}, {'+': lambda a,b: a+b, '-': lambda a,b: a-b}]
	s = ''.join(s.split())
	m = re.findall('(\\+|-|\\*|\\/|[0-9]+(\\.[0-9]+)?|[Pp][Ii])', s)
	exp = []
	for sym in m:
		if sym[0].lower() == 'pi':
			exp.append(np.pi)
		elif re.match('\\+|-|\\*|\\/', sym[0]):
			exp.append(sym[0])
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

parser = argparse.ArgumentParser(description='Converts input file into a fourier series')
parser.add_argument('-i', '--input', type=str, required=True, help='the input file')
parser.add_argument('-o', '--output', type=str, default='out', help='the output file name')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-s', '--svg', action='store_true', help="marks the input file as a svg")
group.add_argument('-b', '--bitmap', action='store_true', help="marks the input file as a bitmap type (bmp, png, jpg, etc.)")
group.add_argument('-v', '--video', action='store_true', help="marks the input file as a video type (mp4, avi, mov, etc.)")
group.add_argument('-p', '--path', action='store_true', help="marks the input file as a .npy (numpy array) file)")

parser.add_argument('-t', '--timescale', type=str, default='1', help='how many seconds video time is 1 second real time (2pi video time seconds is 1 cycle). Accepts math expressions including (+,-,*,/,pi)')
parser.add_argument('-d', '--duration', type=str, default='2*pi', help='the duration of video time to render (2pi video time seconds is 1 cycle). Accepts math expressions including (+,-,*,/,pi)')
parser.add_argument('-tl', '--trail-length', type=str, default='2.1*pi', help='the duration of video time to keep the trail visible (2pi video time seconds is 1 cycle). Accepts math expressions including (+,-,*,/,pi)')
group = parser.add_mutually_exclusive_group()
group.add_argument('-tf', '--trail-fade', action='store_true', help='whether the tail should fade with time')
group.add_argument('-ntf', '--no-trail-fade', action='store_false', help='whether the tail should fade with time')
parser.add_argument('-tc', '--trail-color', type=str, default='#ffff00', help='\'#xxxxxx\' color of the trail as a hexcode')
parser.add_argument('-vc', '--vector-color', type=str, default='#ffffff', help='\'#xxxxxx\'color of the vectors as a hexcode')
parser.add_argument('-fps', type=int, default=60, help='fps of the output video')
parser.add_argument('-fpf', '--frames-per-frame', type=str, default='1', help='A frame is saved every this many frames. There are 2*pi*60/{timescale} frames in a render. Accepts math expressions including (+,-,*,/,pi) casted to int')

parser.add_argument('-dim', '--dimension', type=str, default=None, help='\'[width]x[height]\' dimensions of the output video (defaults to image/video dimensions, or 800x800 for svg, infered using border for path)')
parser.add_argument('--border', type=float, default=0.9, help='percentage (as a float) of border between the path and screen')
group = parser.add_mutually_exclusive_group()
group.add_argument('--density', type=float, default=7, help='how densely packed are samples of a path')
group.add_argument('--points', type=int, default=-1, help='how many point in an image or frame (may be slightly off)')


parser.add_argument('--info', action='store_true', help='Prints some info about the sketch')

parser.add_argument('--save-path', type=str, default=None, help='saves the path in a file to save recomputation (dimensions aren\'t saved)')


args = parser.parse_args()
vColor = int(args.vector_color[1:], base=16)
tColor = int(args.trail_color[1:], base=16)
timescale = strMath(args.timescale)
duration = strMath(args.duration)
trailLength = strMath(args.trail_length)
fpf = int(strMath(args.frames_per_frame))

if args.dimension != None:
	s = ''.join(args.dimension.split()).split('x')
	dims = (int(s[0]/10)*10,int(s[1]/10)*10)
else:
	dims = None

if args.svg:
	if dims == None:
		dims = (800,800)
	path = boundPath(centerPath(svgToPath(args.input, abs(args.density), args.points)), (dims[0]*args.border,dims[1]*args.border))
elif args.bitmap:
	if dims == None:
		size = Image.open(args.input).size
		dims = (int(size[0]/10)*10,int(size[1]/10)*10)
	print('Tracing image')
	path = boundPath(centerPath(imageFileToPath(args.input, abs(args.density), args.points)), (dims[0]*args.border,dims[1]*args.border))
elif args.video:
	path, dims, frames = videoToPath(args.input, abs(args.density), args.points, dims, args.border)
	path = boundPath(centerPath(path), (dims[0]*args.border,dims[1]*args.border))
elif args.path:
	path = np.load(args.input)
	if args.dimension == None:
		dims = (int((max(np.real(path)) - min(np.real(path))) / args.border / 10)*10, int((max(np.imag(path)) - min(np.imag(path))) / args.border / 10)*10)

print()
if args.info:
	print('Number of points:', len(path))
	print('Output dimensions:', dims)
	if args.video:
		print('Input frames:', frames)

if args.save_path != None:
	np.save(args.save_path, path)

renderPath(path, dims, duration, timescale, trailLength, args.trail_fade or args.no_trail_fade, tColor, vColor, args.fps, fpf, args.output)