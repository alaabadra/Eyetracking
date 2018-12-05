from threading import Thread 
class GuiThread(Thread):
	def __init__(self):
		Thread.__init__(self)
	def run(self):
		blink.write_slogan()
