import numpy as np
import scipy.spatial

#https://www.w3.org/Graphics/GIF/spec-gif89a.txt
class GifWriter(object):
	def __init__(self, output):
		self.out = open(output, 'wb')
		self.time = 0
	
	def writeHeader(self, width, height, bpp=4, fps=60):
		self.bpp = max(min(bpp,8),1)
		self.fps = fps
		self.dims = (width, height)
		self.prev = -np.ones((height,width))
		
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
		i = 0
		for c in rgbIntArr:
			self.out.write(int(c).to_bytes(1,byteorder='big'))
			i+=1
		self.pc = np.asarray(rgbIntArr[-6:-3])
		self.vc = np.asarray(rgbIntArr[-3:])
		self.bc = np.asarray(rgbIntArr[:3])
		self.colorVector = np.clip(self.pc-self.bc,-255,255)
	
	def initBackground(self):
		maxLen = 1
		nextCode = 2**self.bpp+2
		dict = {}
		for i in range(2**self.bpp):
			dict[chr(i)] = i
		dict[chr(2**self.bpp)] = 2**self.bpp
		dict[chr(2**self.bpp+1)] = 2**self.bpp+1
		
		##Image Descriptor
		self.out.write(b'\x2C') #Seperator
		self.out.write(b'\x00\x00\x00\x00') #Top Left Corner
		self.out.write(self.dims[0].to_bytes(2,byteorder='little')) #Width
		self.out.write(self.dims[1].to_bytes(2,byteorder='little')) #Height
		self.out.write(b'\x00') #Packed Fields
		
		##Image Data
		self.out.write(self.bpp.to_bytes(1,byteorder='big')) #LZW Minimum Code Size
		
		dataStr = chr(2**self.bpp) + ('\x00'*self.dims[0]*self.dims[1]) + chr(2**self.bpp+1)
		
		ind = 0
		bufferInd = 0
		buffer = bytearray(255)
		acc = 0x0
		accInd = 0
		
		while ind < len(dataStr):
			###########Better search method using forward
			for j in range(maxLen,0,-1):
				if ind+j <= len(dataStr) and dataStr[ind:ind+j] in dict:
					acc += dict[dataStr[ind:ind+j]] << accInd
					accInd += (nextCode-1).bit_length()
					while accInd > 7:
						buffer[bufferInd] = acc & 0xff
						acc = acc >> 8
						accInd -= 8
						bufferInd += 1
						if bufferInd == 0xff:
							bufferInd = 0
							self.out.write(b'\xff')
							self.out.write(buffer)
					if not dataStr[ind:ind+j+1] in dict and ind != 0 and ind != len(dataStr)-1 and nextCode < 2**12:
						dict[dataStr[ind:ind+j+1]] = nextCode
						nextCode += 1
						maxLen = max(maxLen,j+1)
					ind += j
					
					break
		if accInd > 0:
			buffer[bufferInd] = acc & 0xff
			bufferInd += 1
		if bufferInd > 0:
			self.out.write(bufferInd.to_bytes(1,byteorder='big'))
			self.out.write(buffer[:bufferInd])
		self.out.write(b'\x00')
	
	def getDelay(self):
		t = int(self.time-int(self.time)+100/self.fps)
		self.time += 100/self.fps
		return t
	
	def writeFrame(self, image):
		self.writeImage(image, self.getDelay())
	
	# image.shape = (w,h,3)
	def writeImage(self, image, delay):
		image = image[:,:,0:3]
	
		##Graphics Control Extension
		self.out.write(b'\x21\xF9\x04')
		self.out.write(b'\x04') #Packed Fields
		self.out.write(delay.to_bytes(2,byteorder='little')) #Delay time (1/100)
		self.out.write(b'\x00') #Transparent Color Index (not used)
		self.out.write(b'\x00') #Terminator
		
		maxLen = 1
		nextCode = 2**self.bpp+2
		dict = {}
		for i in range(2**self.bpp):
			dict[chr(i)] = i
		dict[chr(2**self.bpp)] = 2**self.bpp
		dict[chr(2**self.bpp+1)] = 2**self.bpp+1
	
	
		np.seterr(divide='ignore',invalid='ignore')
		vcd = np.sum(abs(image - self.vc), axis=2)
		vcd = (vcd-np.linalg.norm(np.subtract(self.vc,self.pc))/2) < 0
		pct = np.sum(( np.clip(image-self.bc,-255,255)/self.colorVector )[:,:,self.colorVector!=0], axis=2)/np.sum(self.colorVector!=0)
		pct = np.round(pct*(2**self.bpp-2))
		
		data = pct
		data[vcd] = 2**self.bpp-1
		
		bound = findBoundary(np.logical_and(abs(data-self.prev)<2,data!=2**self.bpp-1))
		self.prev = data
		
		##Image Descriptor
		self.out.write(b'\x2C') #Seperator
		self.out.write(int(bound[0]).to_bytes(2,byteorder='little'))#Left
		self.out.write(int(bound[1]).to_bytes(2,byteorder='little'))#Top
		self.out.write(int(bound[2]-bound[0]+1).to_bytes(2,byteorder='little'))#Width
		self.out.write(int(bound[3]-bound[1]+1).to_bytes(2,byteorder='little'))#Height
		self.out.write(b'\x00') #Packed Fields
		
		
		##Image Data
		self.out.write(self.bpp.to_bytes(1,byteorder='big')) #LZW Minimum Code Size
		
		dataStr = chr(2**self.bpp) + "".join([chr(item) for item in data[bound[1]:bound[3]+1,bound[0]:bound[2]+1].astype(int).flatten()]) + chr(2**self.bpp+1)
		
		ind = 0
		bufferInd = 0
		buffer = bytearray(255)
		acc = 0x0
		accInd = 0
		
		while ind < len(dataStr):
			###########Better search method using forward
			for j in range(maxLen,0,-1):
				if ind+j <= len(dataStr) and dataStr[ind:ind+j] in dict:
					acc += dict[dataStr[ind:ind+j]] << accInd
					accInd += (nextCode-1).bit_length()
					while accInd > 7:
						buffer[bufferInd] = acc & 0xff
						acc = acc >> 8
						accInd -= 8
						bufferInd += 1
						if bufferInd == 0xff:
							bufferInd = 0
							self.out.write(b'\xff')
							self.out.write(buffer)
					if not dataStr[ind:ind+j+1] in dict and ind != 0 and ind != len(dataStr)-1 and nextCode < 2**12:
						dict[dataStr[ind:ind+j+1]] = nextCode
						nextCode += 1
						maxLen = max(maxLen,j+1)
					ind += j
					
					break
		if accInd > 0:
			buffer[bufferInd] = acc & 0xff
			bufferInd += 1
		if bufferInd > 0:
			self.out.write(bufferInd.to_bytes(1,byteorder='big'))
			self.out.write(buffer[:bufferInd])
		self.out.write(b'\x00')
		
	def close(self):
		self.out.write(b'\x3B')
		self.out.close()
	
#Left Top Right Bottom
def findBoundary(mat):
	vline = np.sum(mat,axis=0)
	hline = np.sum(mat,axis=1)
	return np.argmax(vline>0), np.argmax(hline>0), len(vline) - 1 - np.argmax(vline[::-1]>0), len(hline) - 1 - np.argmax(hline[::-1]>0)
