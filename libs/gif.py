import numpy as np
import scipy.spatial

#https://www.w3.org/Graphics/GIF/spec-gif89a.txt
class GifWriter(object):
	def __init__(self, output):
		self.out = open(output, 'wb')
	
	def writeHeader(self, width, height, bpp=4, fps=60):
		self.bpp = max(min(bpp,8),1)
		self.fps = fps
		
		self.out.write(b'GIF89a')
		self.out.write(width.to_bytes(2,byteorder='little'))
		self.out.write(height.to_bytes(2,byteorder='little'))
		b = 0xf0 + (self.bpp-1)
		self.out.write(b.to_bytes(1,byteorder='big'))
		self.out.write(b'\x00\x00')
	
	def writeApplicationExtensionLoop(self, loopCount=0):
		self.out.write(b'\x21\xFF\x0B')
		self.out.write(b'\x4E\x45\x54\x53\x43\x41\x50\x45\x32\x2E\x30')
		self.out.write(b'\x03\x01\x00\x00')
		self.out.write(b'\x00')
	
	# rgbIntArr = [r0, g0, b0, r1, g1, b1, ...]
	# 0 <= r*,g*,b* <= 255
	def writePalette(self, rgbIntArr):
		palette = np.empty((len(rgbIntArr)//3,3))
		i = 0
		for c in rgbIntArr:
			self.out.write(c.to_bytes(1,byteorder='big'))
			palette[i//3][i%3] = c
			i+=1
		self.paletteTree = scipy.spatial.KDTree(palette,copy_data=True)
	
	def writeFrame(self, image):
		self.writeImage(image, 100//self.fps)
	
	# image.shape = (w,h,3)
	def writeImage(self, image, delay):
		image = image[:,:,0:3]
	
		#Graphics Control Extension
		self.out.write(b'\x21\xF9\x04')
		self.out.write(b'\x00') #Packed Fields
		self.out.write(delay.to_bytes(2,byteorder='little')) #Delay time (1/100)
		self.out.write(b'\x00') #Transparent Color Index (not used)
		self.out.write(b'\x00') #Terminator
		
		#Image Descriptor
		self.out.write(b'\x2C') #Seperator
		self.out.write(b'\x00\x00\x00\x00') #Top Left Corner
		self.out.write(image.shape[1].to_bytes(2,byteorder='little')) #Width
		self.out.write(image.shape[0].to_bytes(2,byteorder='little')) #Height
		self.out.write(b'\x00') #Packed Fields
		
		
		#Image Data
		self.out.write(self.bpp.to_bytes(1,byteorder='big')) #LZW Minimum Code Size
		
		maxLen = 1
		currCode = 2**self.bpp+2
		dict = {}
		for i in range(16):
			dict[chr(i)] = i
		dict[chr(2**self.bpp+1)] = 2**self.bpp
		dict[chr(2**self.bpp+2)] = 2**self.bpp+1
		dataStr = chr(2**self.bpp+1) + "".join([chr(item) for item in self.paletteTree.query(image)[1].flatten()]) + chr(2**self.bpp+2)
		ind = 0
		
		bufferInd = 0
		buffer = bytearray(255)
		acc = 0x0
		accInd = 0
		
		
		
		out = []
		
		
		
		
		while ind < len(dataStr):
			for j in range(maxLen,0,-1):
				if ind+j <= len(dataStr) and dataStr[ind:ind+j] in dict and currCode < 2**12:
					acc += dict[dataStr[ind:ind+j]] << accInd
					accInd += (currCode-1).bit_length()
					out.append(dict[dataStr[ind:ind+j]])
					if accInd > 7:
						buffer[bufferInd] = acc & 0xff
						acc = acc >> 8
						accInd -= 8
						bufferInd += 1
						if bufferInd == 255:
							bufferInd = 0
							self.out.write(b'\xff')
							self.out.write(buffer)
					
					if not dataStr[ind:ind+j+1] in dict and ind != 0 and ind != len(dataStr)-1:
						dict[dataStr[ind:ind+j+1]] = currCode
						currCode += 1
						maxLen = j+1
					ind += j
		if accInd > 0:
			buffer[bufferInd] = acc & 0xff
			bufferInd += 1
		if bufferInd > 0:
			self.out.write(bufferInd.to_bytes(1,byteorder='big'))
			#print(", ".join(hex(b) for b in buffer[:bufferInd]))
			self.out.write(buffer[:bufferInd])
		
		self.out.write(b'\x00')
		
		'''
		print(out)
		print(self.paletteTree.query(image)[1].flatten())
		print([hex(item) for item in self.paletteTree.query(image)[1].flatten()])
		for key in dict:
			print(' '.join([hex(ord(c)) for c in key]),dict[key])
		'''
		
	def close(self):
		self.out.write(b'\x3B')
		self.out.close()


import numpy as np

N=5

gif = GifWriter('test.gif')
gif.writeHeader(N,N)

colors = [0x00,0x00,0x00,0xff,0xff,0xff]
a = np.empty((42))
a[0::3] = np.linspace(255,0,15)[:-1]
a[1::3] = np.linspace(255,0,15)[:-1]
a[2::3] = 0
for b in a.tolist():
	colors.append(int(b))
gif.writePalette(colors)
gif.writeApplicationExtensionLoop()

data = np.empty((N,N,3))

data[:,:,0] = (np.flip(np.linspace(0,1,N))*np.ones((N,1)))*255
data[:,:,1] = (np.flip(np.linspace(0,1,N))*np.ones((N,1)))*255
data[:,:,2] = 0

data = data.astype(int)
gif.writeImage(data,100)
'''
for i in range(N):
	gif.writeImage(data,100)
	data = np.roll(data,1,axis=1)
'''
gif.close();




data = data.astype(np.uint8)
from PIL import Image
im = Image.fromarray(data)
im.save('model.png')
'''
data = np.asarray([np.reshape(colors,(-1,3))]).astype(np.uint8)
im = Image.fromarray(data)
im.save('p.png')
'''
'''
gif = GifWriter('test.gif')
gif.writeHeader(3,5)

colors = [0x00,0x00,0x00,0xff,0xff,0xff]
a = np.empty((42))
a[0::3] = np.linspace(255,0,15)[:-1]
a[1::3] = np.linspace(255,0,15)[:-1]
a[2::3] = 0
for b in a.tolist():
	colors.append(int(b))
gif.writePalette(colors)

gif.writeApplicationExtensionLoop()

data = np.asarray([[[0x00,0x00,0x00],[0xff,0xff,0xff],[0xff,0xff,0xff]] , [[0xff,0xff,0xff],[0x00,0x00,0x00],[0xff,0xff,0xff]] , [[0xff,0xff,0x00],[0xdd,0xdd,0x00],[0xbb,0xbb,0x00]] , [[0x99,0x99,0x00],[0x77,0x77,0x00],[0x55,0x55,0x00]] , [[0x33,0x33,0x00],[0x11,0x11,0x00],[0x00,0x00,0x00]]])
gif.writeImage(data,100)

data = np.asarray([[[0xff,0xff,0xff],[0x00,0x00,0x00],[0x00,0x00,0x00]] , [[0x00,0x00,0x00],[0xff,0xff,0xff],[0x00,0x00,0x00]] , [[0xff,0xff,0x00],[0xdd,0xdd,0x00],[0xbb,0xbb,0x00]] , [[0x99,0x99,0x00],[0x77,0x77,0x00],[0x55,0x55,0x00]] , [[0x33,0x33,0x00],[0x11,0x11,0x00],[0x00,0x00,0x00]]])
gif.writeImage(data,100)

gif.close();
'''