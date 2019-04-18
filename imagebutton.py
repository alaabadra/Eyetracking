from tkinter import *
root = Tk()
root['bg'] = white
okbutton =PhotoImage(file='door.png')
a = Button(root, image=okbutton,bg = 'white')
a['border'] = '0'
a.grid()

root.mainloop()