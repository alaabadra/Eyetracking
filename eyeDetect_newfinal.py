import cv2
import numpy as np
import dlib
import pyautogui, sys, pytweening
import blink
import tkinter as tk




# returns center in form (y,x)
def featureCenterXY(rect):
    #eyes are arrays of the form [minX, minY, maxX, maxY]
    return (.5*(rect[0]+rect[2]), .5*(rect[1]+rect[3]))

def contains(outerFeature, innerFeature):
    p = featureCenterXY(innerFeature)
    #eyes are arrays of the form [minX, minY, maxX, maxY]
    return p[0] > outerFeature[0] and p[0] < outerFeature[2] and p[1] > outerFeature[1] and p[1] < outerFeature[3]


def getLeftAndRightEyes(faces, eyes):

    if len(eyes)==0:
        return ()
    for face in faces:
        for i in range(eyes.shape[0]):
            for j in range(i+1,eyes.shape[0]):
                leftEye = eyes[i] #by left I mean camera left
                rightEye = eyes[j]
                #eyes are arrays of the form [minX, minY, maxX, maxY]
                if (leftEye[0]+leftEye[2]) > (rightEye[0]+rightEye[2]): #leftCenter is > rightCenter
                    rightEye, leftEye = leftEye, rightEye #swap
                if contains(leftEye,rightEye) or contains(rightEye, leftEye):#they overlap. One eye containing another is due to a double detection; ignore it
                    debugPrint('rejecting double eye')
                    continue
                if leftEye[3] < rightEye[1] or rightEye[3] < leftEye[1]:#top of one is below (>) bottom of the other. One is likely a mouth or something, not an eye.
                    debugPrint('rejecting non-level eyes')
                    continue

                if not (contains(face,leftEye) and contains(face,rightEye)):#face contains the eyes. This is our standard of humanity, so capture the face.
                    debugPrint("face doesn't contain both eyes")
                    continue
                
                return (leftEye, rightEye)

    return ()

verbose=True

def debugPrint(s):
    if verbose:
        print(s)

showMainImg=True;

def debugImg(arr):
    global showMainImg
    showMainImg=False;
    toShow = cv2.resize((arr-arr.min())*(1.0/(arr.max()-arr.min())),(0,0), fx=8,fy=8,interpolation=cv2.INTER_NEAREST)
    cv2.imshow(WINDOW_NAME,toShow)

BLOWUP_FACTOR = 1 # Resizes image before doing the algorithm. Changing to 2 makes things really slow. So nevermind on this.
RELEVANT_DIST_FOR_CORNER_GRADIENTS = 8*BLOWUP_FACTOR
dilationWidth = 1+2*BLOWUP_FACTOR #must be an odd number
dilationHeight = 1+2*BLOWUP_FACTOR #must be an odd number
dilationKernel = np.ones((dilationHeight,dilationWidth),'uint8')





writeEyeDebugImages = False #enable to export image files showing pupil center probability
eyeCounter = 0

def getPupilCenter(gray, getRawProbabilityImage=False):

    gray = gray.astype('float32')
    if BLOWUP_FACTOR != 1:
        gray = cv2.resize(gray, (0,0), fx=BLOWUP_FACTOR, fy=BLOWUP_FACTOR, interpolation=cv2.INTER_LINEAR)

    IRIS_RADIUS = gray.shape[0]*.75/2
    dxn = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3) 
    dyn = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    magnitudeSquared = np.square(dxn)+np.square(dyn)

    magThreshold = magnitudeSquared.mean()*.6 
    
    gradsTouse = (magnitudeSquared>magThreshold) & (np.abs(4*dxn)>np.abs(dyn))
    lengths = np.sqrt(magnitudeSquared[gradsTouse]) #this converts us to double format
    gradDX = np.divide(dxn[gradsTouse],lengths) #unrolled columnwise
    gradDY = np.divide(dyn[gradsTouse],lengths)

    isDark = gray< (gray.mean()*.8)  #<-- TUNABLE PARAMETER
    global dilationKernel
    isDark = cv2.dilate(isDark.astype('uint8'), dilationKernel) #dilate so reflection goes dark too
    gradXcoords =np.tile( np.arange(dxn.shape[1]), [dxn.shape[0], 1])[gradsTouse]
    gradYcoords =np.tile( np.arange(dxn.shape[0]), [dxn.shape[1], 1]).T[gradsTouse] 
    minXForPupil = 0 #int(dxn.shape[1]*.3)

    centers = np.array([[phiWithHist(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS) if isDark[cy][cx] else 0 for cx in range(minXForPupil,dxn.shape[1])] for cy in range(dxn.shape[0])]).astype('float32')
    maxInd = centers.argmax()
    (pupilCy,pupilCx) = np.unravel_index(maxInd, centers.shape)
    pupilCx += minXForPupil
    pupilCy /= BLOWUP_FACTOR
    pupilCx /= BLOWUP_FACTOR
    if writeEyeDebugImages:
        global eyeCounter
        eyeCounter = (eyeCounter+1)%5 #write debug image every 5th frame
        if eyeCounter == 1:
            cv2.imwrite( "eyeGray.png", gray/gray.max()*255) #write probability images for our report
            cv2.imwrite( "eyeIsDark.png", isDark*255)
            cv2.imwrite( "eyeCenters.png", centers/centers.max()*255)
    if getRawProbabilityImage:
        return (pupilCy, pupilCx, centers)
    else:
        return (pupilCy, pupilCx)


def phiWithHist(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS):
    vecx = gradXcoords-cx
    vecy = gradYcoords-cy
    lengthsSquared = np.square(vecx)+np.square(vecy)
    # bin the distances between 1 and IRIS_RADIUS. We'll discard all others.
    binWidth = 1 #TODO: account for webcam resolution. Also, maybe have it transform ellipses to circles when on the sides? (hard)
    numBins =  int(np.ceil((IRIS_RADIUS-1)/binWidth))
    bins = [(1+binWidth*index)**2 for index in range(numBins+1)] #express bin edges in terms of length squared
    hist = np.histogram(lengthsSquared, bins)[0]
    maxBin = hist.argmax()
    slop = binWidth
    valid = (lengthsSquared > max(1,bins[maxBin]-slop)) &  (lengthsSquared < bins[maxBin+1]+slop)
    dotProd = np.multiply(vecx,gradDX)+np.multiply(vecy,gradDY)
    valid = valid & (dotProd > 0) 
    dotProd = np.square(dotProd[valid]) # dot products squared
    dotProd = np.divide(dotProd,lengthsSquared[valid]) #make normalized squared dot products

    dotProd = np.square(dotProd) # squaring puts an even higher weight on values close to 1
    return np.sum(dotProd) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2

def phiCorner(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords):
    vecx = gradXcoords-cx
    vecy = gradYcoords-cy
    angles = np.arctan2(vecy,vecx)
    lengthsSquared = np.square(vecx)+np.square(vecy)
    valid = (lengthsSquared > 0) & (lengthsSquared < RELEVANT_DIST_FOR_CORNER_GRADIENTS) & (vecx>0.4)#RIGHT EYE ASSUMPTION
    numBins = 10

    (hist,bins) = np.histogram(angles, numBins, (-math.pi,math.pi))
    slop = math.pi/numBins/2
    maxBin = hist.argmax()
    hist[maxBin] = 0;
    hist[max(0,maxBin-1)]=0;
    hist[min(maxBin+1,numBins-1)]=0;
    secondMaxBin = hist.argmax();
    stat = angles #gradDY
    validBina = valid & ((bins[maxBin]-slop<stat)&(stat<bins[maxBin+1]+slop))
    validBinb = valid & ((bins[secondMaxBin]-slop<stat)&(stat<bins[secondMaxBin+1]+slop))#use only points near the histogram max


    dotProd = np.multiply(vecx,gradDX)+np.multiply(vecy,gradDY)
    dotProda = np.square(dotProd[validBina]) # dot products squared
    dotProdb = np.square(dotProd[validBinb]) # dot products squared
    dotProda = 1.0-np.divide(dotProda,lengthsSquared[validBina])
    dotProdb = 1.0-np.divide(dotProdb,lengthsSquared[validBinb]) #make normalized squared dot products, and take 1-them so 0 gets the 
    dotProda = np.square(dotProda) #only count dot products that are really close
    dotProdb = np.square(dotProdb) #only count dot products that are really close
    suma = np.sum(dotProda) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2
    sumb = np.sum(dotProdb) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2
    return min(suma,sumb)+.5*max(suma,sumb) #this score should favor a strong bimodal histogram shape

def detect(img, cascade, minimumFeatureSize=(20,20)):
    if cascade.empty():
        raise(Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2] #convert last coord from (width,height) to (maxX, maxY)
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

# init the filters we'll use below
haarFaceCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
haarEyeCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_eye.xml")


def getOffset(frame, allowDebugDisplay=True, trackAverageOffset=True, directInferenceLeftRight=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = gray.copy()
    gray = cv2.equalizeHist(gray)
    x0 = 768  # height your screen
    y0 = 1366  # Width your screen
    threshold_size_X = 13
    threshold_size_Y = 15
    x,y,_ = frame.shape
    # find faces and eyes
    minFaceSize = (80,80)
    minEyeSize = (25,25)
    faces = detect(gray,haarFaceCascade,minFaceSize)
    eyes = detect(gray,haarEyeCascade,minEyeSize)
    drawKeypoints = allowDebugDisplay #can set this false if you don't want the keypoint ID numbers
    if allowDebugDisplay:
        output = frame
        draw_rects(output,faces,(0,255,0)) #BGR format
    else:
        output = None

    leftEye_rightEye = getLeftAndRightEyes( faces, eyes)
    if leftEye_rightEye: #if we found valid eyes in a face
        xDistBetweenEyes = (leftEye_rightEye[0][0]+leftEye_rightEye[0][1]+leftEye_rightEye[1][0]+leftEye_rightEye[1][1])/4 
        pupilXYList = []
        pupilCenterEstimates = []
        cen = []
        for eyeIndex, eye in enumerate(leftEye_rightEye):

            corner = eye.copy()
            eyeWidth = eye[2]-eye[0]
            eyeHeight = eye[3]-eye[1]
            eye[0] += eyeWidth*.20
            eye[2] -= eyeWidth*.15
            eye[1] += eyeHeight*.3
            eye[3] -= eyeHeight*.2
            eye = np.round(eye)
            eyeImg = gray[eye[1]:eye[3], eye[0]:eye[2]]
            if directInferenceLeftRight:
                (cy,cx, centerProb) = getPupilCenter(eyeImg, True)
                pupilCenterEstimates.append(centerProb.copy())
            else:
                (cy,cx) = getPupilCenter(eyeImg, True)
            pupilXYList.append( (cx+eye[0],cy+eye[1])  )
            if len(cen) == 2:
               cen =[]
            else:
               cen.append([int(pupilXYList[eyeIndex][0]), int(pupilXYList[eyeIndex][1])])
          
            if allowDebugDisplay:
                if len(cen) == 2:
                   data = np.array(cen)
                   center = np.average(data, axis=0)
                   
                   cv2.circle(output, (int(center[0]), int(center[1])), 3, (255,255,0),thickness=1) 
                   nn ,n = center[0], center[1]

                   o = (n/y) * 100.0
                   ox = (nn/x) * 100.0
                   
                   ob = ((y0+50)/100.0) * o
                   obx = ((x0+50)/100.0) * ox
                   
                   ob = int(ob)
                   obx = int(obx)
                   #print((ob, obx),'::::' ,n,nn,'y,x,y0,x0_',y,x,y0,x0)
                   if ob !=0 and obx !=0:
                      pyautogui.moveTo(obx, ob)
                   
                cv2.rectangle(output, (eye[0], eye[1]), (eye[2], eye[3]), (0,255,0), 1)
                #print(pupilXYList[eyeInadex])
                cv2.circle(output, (int(pupilXYList[eyeIndex][0]), int(pupilXYList[eyeIndex][1])), 3, (255,0,0),thickness=1) #BGR format
                #print((int(pupilXYList[eyeIndex][0]), int(pupilXYList[eyeIndex][1])))


                width, height = pyautogui.size()

                ###############----------new block -----------#############

                face = faces[0]
                face_one = dlib.rectangle(face[0], face[1],face[2 ], face[3])
                
                stat1 = blink.blinks(gray2 ,face_one)
                print(stat1)
                stat2 = blink.write_slogan()
                print(stat2)
                
                #stat2 = blink.write_slogan()
                #print(stat2)
                
               #stat2 = b1.write_slogan()
                #print(stat2)
                selected=0
                if stat1 == 'closed_eye':
                    cv2.putText(output, "eyes: {}".format(stat1), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 244, 255), 2)
                    if((int(pupilXYList[eyeIndex][0]))>=0 and (int(pupilXYList[eyeIndex][0]))<325):
                        
                           
                        selected=1
                        button1 = tk.Button(activebackground='yellow')
                        button1.pack(side=tk.LEFT)
                        print('button11')
                        #root.mainloop()   
                    if((int(pupilXYList[eyeIndex][0]))>=330 and (int(pupilXYList[eyeIndex][0]))<655):
                        
                           
                        selected=1
                        button2 = tk.Button(activebackground='yellow')
                        button2.pack(side=tk.LEFT)
                        print('button22')
                        #root.mainloop()   
                    if((int(pupilXYList[eyeIndex][0]))>=660 and (int(pupilXYList[eyeIndex][0]))<985):
                        
                       selected=1
                       button3 = tk.Button(activebackground='yellow')
                       button3.pack(side=tk.LEFT)
                       print('button33')
                       #root.mainloop()   
                    else:
                        
                        selected=1
                        button4 = tk.Button(activebackground='yellow')
                        button4.pack(side=tk.LEFT)
                        print('button44')
                        #root.mainloop()   

                                                
                if stat1 == 'open_eye':
                    #def createLayout(self):
                            #self.SetTitle("enable/disable")
                    #stat2 = blink.write_slogan()
                    cv2.putText(output, "eyes: {}".format(stat1), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 244, 255), 2)   
                    if((int(pupilXYList[eyeIndex][0]))>=0 and (int(pupilXYList[eyeIndex][0]))<340):
                            #x=(int(pupilXYList[eyeIndex][0]))>=0 and (int(pupilXYList[eyeIndex][0]))<340
                            #self.AddButton(1011, c4d.BFH_MASK, initw=145, name="Enable")
                            #pyautogui.click()
                        #button1 = tk.Button(activebackground='green')
                        #button1.pack(side=tk.LEFT)
                        button1 = tk.Button(activebackground='green')
                        button1.pack(side=tk.LEFT)
                        button2 = tk.Button(activebackground='red')
                        button2.pack(side=tk.LEFT)
                        button3 = tk.Button(activebackground='red')
                        button3.pack(side=tk.LEFT)
                        button4 = tk.Button(activebackground='red')
                        button4.pack(side=tk.LEFT)
                        #root.mainloop()   
                    elif((int(pupilXYList[eyeIndex][0]))>=340 and (int(pupilXYList[eyeIndex][0]))<680):
                            #pyautogui.doubleClick()
                        #slogan = tk.Button(activebackground='green')
                        #slogan.pack(side=tk.LEFT)
                        button1 = tk.Button(activebackground='red')
                        button1.pack(side=tk.LEFT)
                        button2 = tk.Button(activebackground='green')
                        button2.pack(side=tk.LEFT)
                        button3 = tk.Button(activebackground='red')
                        button3.pack(side=tk.LEFT)
                        button4 = tk.Button(activebackground='red')
                        button4.pack(side=tk.LEFT)
                        
                        #root.mainloop()   
                    elif((int(pupilXYList[eyeIndex][0]))>=680 and (int(pupilXYList[eyeIndex][0]))<1020):
                            #pyautogui.tripleClick()
                        #slogan2 = tk.Button(activebackground='green')
                        #slogan2.pack(side=tk.LEFT)
                        button1 = tk.Button(activebackground='red')
                        button1.pack(side=tk.LEFT)
                        button2 = tk.Button(activebackground='red')
                        button2.pack(side=tk.LEFT)
                        button3 = tk.Button(activebackground='green')
                        button3.pack(side=tk.LEFT)
                        button4 = tk.Button(activebackground='red')
                        button4.pack(side=tk.LEFT)
                        #root.mainloop()   
                    else:

                            #pyautogui.click()
                        #slogan3 = tk.Button( activebackground='green' )
                        #slogan3.pack(side=tk.LEFT)
                        button1 = tk.Button(activebackground='red')
                        button1.pack(side=tk.LEFT)
                        button2 = tk.Button(activebackground='red')
                        button2.pack(side=tk.LEFT)
                        button3 = tk.Button(activebackground='red')
                        button3.pack(side=tk.LEFT)
                        button4 = tk.Button(activebackground='green')
                        button4.pack(side=tk.LEFT)
                        #root.mainloop()   
                
                
                    pytweening.linear(0.75)
                    ######### this code to determine x,y
                    x, y = pyautogui.position()
                    positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
                    print(positionStr, end='')
                    print('\b' * len(positionStr), end='', flush=True)
            #root.mainloop()

    cv2.imshow(WINDOW_NAME,output)

WINDOW_NAME = "preview"
def main():
    cv2.namedWindow(WINDOW_NAME) # open a window to show debugging images

    vc = cv2.VideoCapture(0) # Initialize the default camera
    try:
        if vc.isOpened(): # try to get the first frame
            (readSuccessful, frame) = vc.read()
            #frame = cv2.flip(frame,1)
            print(frame.shape)
        else:
            raise(Exception("failed to open camera."))
            readSuccessful = False
    
        while readSuccessful:
            getOffset(frame, allowDebugDisplay=True)
            key = cv2.waitKey(10)
            if key == ord('q'): # exit on ESC
                cv2.imwrite( "lastOutput.png", frame) #save the last-displayed image to file, for our report
                break
            # Get Image from camera
            readSuccessful, frame = vc.read()
    finally:
        vc.release() #close the camera
        cv2.destroyWindow(WINDOW_NAME) #close the window


if __name__ == '__main__':
    main()

