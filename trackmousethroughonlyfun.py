#! python3
import pyautogui, sys, pytweening
print('Press Ctrl-C to quit.')
try:
    while True:
        #pyautogui.moveTo(100, 200)  # moves mouse to X of 100, Y of 200.
        # moves mouse to X of 263, Y of 187,,,righttop,,,change with 'rightbottom' by y almost 5 and x must 0 but here 2,,,and change with 'lefttop' by x almost 3 and x must 0 but here 3.


        #pyautogui.moveTo(270, 160)  # ,,left top ,,,,change with 'leftbottom' by y almost 30 and x must 0 but here 1,,,and change with 'righttop' by x almost 11 and y must 0 but here 15.



        #pyautogui.moveTo(271, 190)  # ,,left bottom ,,,,change with 'rightbottom' by x almost 10  and x must 0 but here 2,,,

        #pyautogui.moveTo(259, 186)  # ,,right top ,,,,change with 'rightbottom' by y almost 6 and x must 0 but here 2,,,and change with 'lefttop' by x almost 11 and y must 0 but here 15.

        #pyautogui.moveTo(261, 192)  # ,,right bottom 
        
        #pyautogui.size()#(1920, 1080)
        pyautogui.position()#(187, 567)
        #pyautogui.onScreen(0, 0)#true
        #pyautogui.onScreen(0, -1)#false
        ########Mouse Movement
        #pyautogui.moveTo(100, 200)  # moves mouse to X of 100, Y of 200.
        #pyautogui.moveTo(None, 500)## moves mouse to X of 100, Y of 500.
        #pyautogui.moveTo(100, 200, pyautogui.MINIMUM_DURATION)# moves mouse to X of 100, Y of 200 pyautogui.MINIMUM_DURATION  is 0.1.
        #pyautogui.moveTo(100, 200, 0.5)# moves mouse to X of 100, Y of 200 0.5
        #pyautogui.moveRel(-30, 0)     # move the mouse left 30 pixels.
        ########Mouse Drags
        #pyautogui.dragTo(100, 200, button='left')     # drag mouse to X of 100, Y of 200 while holding down left mouse button
        #pyautogui.dragTo(300, 400, 2, button='left')  # drag mouse to X of 300, Y of 400 over 2 seconds while holding down left mouse button
        #########Tween / Easing Functions
        #pyautogui.moveTo(100, 100, 2, pyautogui.easeInQuad)     # start slow, end fast
        #pyautogui.moveTo(100, 100, 2, pyautogui.easeInOutQuad)  # start and end fast, slow in middle
        #######Mouse Clicks
        #pyautogui.click()  # click the mouse
        #pyautogui.click(x=100, y=200)  # move to 100, 200, then click the left mouse button.
        #pyautogui.click(button='right')  # right-click the mouse
        #pyautogui.click(clicks=2)  # double-click the left mouse button
        #pyautogui.click(clicks=2, interval=0.25)  # double-click the left mouse button, but with a quarter second pause in between clicks
        #pyautogui.click(button='right', clicks=3, interval=0.25)  ## triple-click the right mouse button with a quarter second pause in between clicks
        #pyautogui.doubleClick()  # perform a left-button double click
        #pyautogui.tripleClick()
        ##########The mouseDown() and mouseUp() Functions
        #pyautogui.mouseDown(); pyautogui.mouseUp()  # does the same thing as a left-button mouse click
        #pyautogui.mouseDown(button='right')  # press the right button down
        #pyautogui.mouseUp(button='right', x=100, y=200)  # move the mouse to 100, 200, then release the right button up.
        ##########Mouse Scrolling
       # pyautogui.scroll(10)   # scroll up 10 "clicks"
        #pyautogui.hscroll(-10)   # scroll left 10 "clicks"
        pytweening.linear(0.75)
        ######### this code to determine x,y
        x, y = pyautogui.position()
        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
        print(positionStr, end='')
        print('\b' * len(positionStr), end='', flush=True)
except KeyboardInterrupt:
    print('\n')
