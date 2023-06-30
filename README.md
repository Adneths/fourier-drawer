# fourier-drawer
<div align="center">
<img src="./assets/title.gif" width="95%">
</div>

## Requires
- Python3
- ffmpeg [download](https://ffmpeg.org/download.html)

## Installation
1. Install the above requirements
2. Clone/download this repository
3. Run `pip install -r requirements.txt`

## Usage
```
python main.py [-h] -i INPUT (-s | -b | -v | -p) [--density DENSITY | --points POINTS] [-o OUTPUT] [-dim DIMENSION]
               [-fps FPS] [--save-path SAVE_PATH] [-t TIMESCALE] [-d DURATION] [-ss START] [-tl TRAIL_LENGTH]
               [-tf | -ntf] [-tc TRAIL_COLOR] [-vc VECTOR_COLOR] [-tw TRAIL_WIDTH] [-vw VECTOR_WIDTH]
               [-fpf FRAMES_PER_FRAME] [--center CENTER] [-view VIEWPORT] [-z ZOOM] [-ft] [-g [GPU]] [--info INFO]
               [--profile]

  -h, --help				show this help message and exit

Input Parameter:
  -i INPUT, --input INPUT		the input file
  -s, --svg             		marks the input file as a svg
  -b, --bitmap          		marks the input file as a bitmap type (bmp, png, jpg, etc.)
  -v, --video           		marks the input file as a video type (mp4, avi, mov, etc.)
  -p, --path            		marks the input file as a .npy (numpy array) file)
  --density DENSITY     		how densely packed are samples of a path
  --points POINTS       		how many point in an image or frame

Output Parameter:
  -o OUTPUT, --output OUTPUT		the output file name
  -dim [w]x[h], --dimension [w]x[h]	dimensions of the input (defaults to image/video dimensions,
						or 800x800 for svg)
  -fps FPS              		fps of the output video
  --save-path SAVE_PATH			saves the path in a file to save recomputation

Render Parameter:
  -t {math}, --timescale {math}		how many seconds video time is 1 second real time
  -d {math}, --duration {math}		the duration of video time to write to file
  -ss {math}, --start {math}		the time after which writing to file begins
  -tl {math}, --trail-length {math}	the duration of video time to keep the trail visible
  -tf, --trail-fade     		whether the tail should fade with time
  -ntf, --no-trail-fade			whether the tail should fade with time
  -tc TRAIL_COLOR, --trail-color TRAIL_COLOR
                        		'#xxxxxx' color of the trail as a hexcode
  -vc VECTOR_COLOR, --vector-color VECTOR_COLOR
                        		'#xxxxxx' color of the vectors as a hexcode
  -tw TRAIL_WIDTH, --trail-width TRAIL_WIDTH
                        		width of the trail
  -vw VECTOR_WIDTH, --vector-width VECTOR_WIDTH
                        		width of the vectors
  -fpf {math}, --frames-per-frame {math}
                       			one video frame is saved every this many `dt`s.
					There are 2*pi*60/{timescale} frames in a render. (Casted to int)
  --center [x]x[y]       		offset from the center
  -view [w]x[h], --viewport [w]x[h]	dimensions of the output video (defaults to image/video dimensions,
						or 800x800 for svg)
  -z ZOOM, --zoom ZOOM  		percentage (as a float) of border between the path and screen
  -ft, --follow-trail   		centers video on the head of vectors (includs offset)
  -g [GPU], --gpu [GPU]			use Cuda to accelerate rendering process (use a number to specify a GPU
						or ? to list avaliable GPUs)

Debug Parameter:
  --info [INFO]           		d for Debug, p for Path, r for Render, g for GPU
  --profile             		profiles timing information


Note:
  - There are 2pi video time seconds in 1 cycle
  - {math} accepts mathematical expressions (ex. '2*pi+1', '1/${frames}') 
    Valid symbols: any number using [0-9.], +-*/, pi, ${frames}
```

### Multi-render
Render multiple perspectives of the same series by specifying a `*.json` and passing it to the `-o` flag.
#### `*.json` format:
```
{
	"renders": [
		{
			"x": 0,
			"y": 0,
			"width": 800,
			"height": 800,
			"zoom": 0.9,
			"vector_width": 1,
			"trail_width": 1,
			"vector_color": "#ffffff",
			"trail_color": "#ffff00",
			"fps": 60,
			"output": "out.mp4",
			"follow_path": false,
			"trail_fade": true
		},
		<other render perspectives>
	]
}
```

## Examples

<div align="center">
<img src="./assets/e0.gif" width="400vw">
</div>

```
python main.py -i sample\yt.svg -s -o sample\yt -ss 2*pi --density 5 -dim 600x400
```

<div align="center">
<img src="./assets/e1.gif" width="350vw">
<img src="./assets/e2.gif" width="350vw">
</div>

```
python main.py -i sample\chrome.svg -s -o sample\chrome -t 1/8 -fpf 4 --density 25 -ss 2*pi -tl 1.5*pi
python main.py -i sample\grass.png -b -o sample\grass -t 1/8 -fpf 4 -d 3*pi -dim 400x400 -tl 2.5*pi -tc #00ff00 -vc #888888
```

<div align="center">
<img src="./assets/e3.gif" width="800vw">
</div>

```
python main.py -i sample\zvs.png -b -o sample\zvs -t 1/64 -fpf 24 -d 2*pi -ntf
```
