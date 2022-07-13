# fourier-drawer
<div align="center">
<img src="./assets/title.gif" width="95%">
</div>

## Requires
- ffmpeg [download](https://ffmpeg.org/download.html)

## Usage
```
python main.py [-h] -i INPUT [-o OUTPUT] (-s | -b | -v | -p) [-t TIMESCALE] [-d DURATION]
			   [-tl TRAIL_LENGTH] [-tf | -ntf] [-tc TRAIL_COLOR] [-vc VECTOR_COLOR] [-fps FPS]
			   [-fpf FRAMES_PER_FRAME] [-dim DIMENSION] [--border BORDER]
			   [--density DENSITY | --points POINTS] [--info] [--save-path SAVE_PATH]

  -h, --help				show this help message and exit

Input Options
  -i INPUT, --input INPUT		the input file
  -o OUTPUT, --output OUTPUT		the output file name
  
  -s, --svg             		marks the input file as a svg
  -b, --bitmap          		marks the input file as a bitmap type (bmp, png, jpg, etc.)
  -v, --video           		marks the input file as a video type (mp4, avi, mov, etc.)
  -p, --path            		marks the input file as a .npy (numpy array) file)

Rendering Options
  -t {math}, --timescale {math} 	how many seconds video time is 1 second real time
  -d {math}, --duration {math} 		the duration of video time to render
  -tl {math}, --trail-length {math} 	the duration of video time to keep the trail visible
  -tf, --trail-fade     		whether the tail should fade with time (default)
  -ntf, --no-trail-fade 		whether the tail should fade with time
  -tc #xxxxxx, --trail-color #xxxxxx 	color of the trail as a hexcode
  -vc #xxxxxx, --vector-color #xxxxxx 	color of the vectors as a hexcode
  -fps FPS              		fps of the output video
  -fpf {math}, --frames-per-frame {math}
					A frame is saved every this many frames.
					There are 2*pi*60/{timescale} frames in a render. (Casted to int)
								
  -dim [w]x[h], --dimension [w]x[h] 	dimensions of the output video (defaults to image/video dimensions,
					or 800x800 for svg, infered using border for path)

  --border BORDER       		percentage (as a float) of border between the path and screen
  
Path Options
  --density DENSITY     		how densely packed are samples of a path
  --points POINTS       		how many point in an image or frame (may be slightly off)

Miscellaneous Options
  --info                		Prints some info about the sketch
  --save-path SAVE_PATH 		saves the path in a file to save recomputation
					(dimensions aren't saved but infered on load)

Note:
  - There are 2pi video time seconds in 1 cycle
  - {math} accepts mathematical expressions (ex. '2*pi+1', '1/${frames}') 
    Valid symbols: any number using [0-9.], +-*/, pi, ${frames}
```

## Examples

<div align="center">
<img src="./assets/e0.gif" width="400vw">
</div>

```
python main.py -i sample\yt.svg -s -o sample\yt -t 1/2 -fpf 1 -d 3*pi --density 15 -dim 900x600
```

<div align="center">
<img src="./assets/e1.gif" width="350vw">
<img src="./assets/e2.gif" width="350vw">
</div>

```
python main.py -i sample\chrome.svg -s -o sample\chrome -t 1/8 -fpf 2 -d 3*pi --density 15 -tl 2.5*pi
python main.py -i sample\grass.png -b -o sample\grass -t 1/8 -fpf 2 -d 3*pi -tl 2.5*pi -tc #00ff00 -vc #888888
```

<div align="center">
<img src="./assets/e3.gif" width="800vw">
</div>

```
python main.py -i sample\zvs.png -b -o sample\zvs -t 1/64 -fpf 16 -d 3*pi -ntf
```
