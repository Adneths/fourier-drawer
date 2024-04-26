def printProgressBar(percentage, prefix = '', suffix='', decimals = 1, length = 40):
	percent = ("{0:." + str(decimals) + "f}").format(100*percentage)
	fill = int(length * percentage)
	bar = '*' * fill + '-' * (length - fill)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = '')

import time
class EstimateTimeRemaining:
	def __init__(self):
		self.pT = time.time()
		self.s = -1
		self.prevS = -1
		self.string = 'XX:XX'
		self.d60 = [0]*60
		self.tail = 0
		self.t = time.time()
	
	def sample(self, samplesLeft):
		self.d60[self.tail] = time.time()-self.t
		self.t = time.time()
		self.tail = (self.tail+1)%60
		if time.time()-self.pT > 2:
			self.pT = time.time()
			self.prevS = self.s
			self.s = samplesLeft*sum(self.d60)/60
		
	def seconds_remaining(self):
		return self.s
	
	def formatted_seconds_remaining(self):
		if self.prevS != self.s:
			if self.s < 0:
				self.string = 'XX:XX'
			elif self.s < 3600:
				self.string = '{:02}:{:02.0f}'.format(int((self.s%3600)/60),self.s%60)
			else:
				self.string = '{}:{:02}:{:02.0f}'.format(int(self.s/3600),int((self.s%3600)/60),self.s%60)
		return self.string