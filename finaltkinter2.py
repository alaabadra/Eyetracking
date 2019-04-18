import Tkinter as tk
import os.path

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
def _abspath(name):
    return os.path.abspath(os.path.join(PROJECT_DIR, name))


class Application(tk.Frame):
    def __init__(self, master=None, background_color='#000000', background_image=None):
        tk.Frame.__init__(self, master, background=background_color)
        self.images = {}
        self.buttons = []
        self.pressed = None
        self.background_color = background_color
        self.background_image = self._image(background_image) if background_image else None
        self.master.configure(bg=background_color)
        self.place(relx=0.5, rely=0.5, anchor='c')
        self._create_widgets()

    def _image(self, name):
        if name in self.images:
            return self.images[name]
        else:
            path = _abspath(name)
            if not os.path.exists(path):
                return None
            # print 'Loading image from', path
            image = tk.PhotoImage(file=path)
            self.images[name] = image
            return image

    def _action(self, name):
        # print 'action:', name
        if name == 'quit-button':
            self.quit()

    def _button(self, name, width=None, height=None, bd=0, style=None, **kwargs):
        image = self._image(name + '.gif')
        if not image:
            print 'Button image not found:', name+'.gif'
        label = tk.Label(
            self,
            image=image,
            relief='flat',
            bd=0,
            background=self.background_color,
            **kwargs
        )

        # Update the image when pressed
        pressed_image = self._image(name+'-pressed.gif') or image

        def _press(evt=None):
            self.pressed = label
            label.configure(image=pressed_image)

        label.bind("<ButtonPress-1>", _press)

        def _leave(evt=None):
            if self.pressed is label:
                label.configure(image=image)
                self.pressed = None

        label.bind('<Leave>', _leave)

        def _release(evt=None):
            # Should only do this if we didn't move the mouse away
            if self.pressed is label:
                _leave(evt)
                self._action(name)

        label.bind("<ButtonRelease-1>", _release)

        return label

    def _create_widgets(self):
        # Background image - only works if your buttons are square without transparency, since this
        # toolkit doesn't support transparent buttons.
        if self.background_image:
            self.background_label = tk.Label(self, image=self.background_image)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.buttons = [
            self._button('button1'),
            self._button('button2'),
            self._button('button3'),
            self._button('quit-button', height=400),  # Example of setting height per button
        ]
        for button in self.buttons:
            button.grid()  # Add button to the UI


app = Application(
    background_color='#000000',
    background_image='background.gif'
)
app.master.title('Menu')
app.master.overrideredirect(True)  # Remove window frame / buttons
app.master.geometry("{0}x{1}+0+0".format(app.master.winfo_screenwidth(), app.master.winfo_screenheight())) # Full screen
app.mainloop()