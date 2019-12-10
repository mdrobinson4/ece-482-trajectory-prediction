import numpy as np
import cv2
import time
import math
import json

ballWidth = 749.3 # width in mm
squareWidth = 609.6
rimWidth = 457.2
score = False
yA = 0
side = 'r'

def main():
    trajFlag = False
    cap = cv2.VideoCapture("./tuv/4_Trim.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fgbg = cv2.createBackgroundSubtractorMOG2(500, 16, True)

    rimPosition = {}
    hsvRange = getHSV('hsvRange.json')
    trajectory = {'x': [], 'y': [], 'z': []}
    state = {'position': [], 'radius': [], 'velocity': []}
    dt = 1 / fps
    frameCount = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        frame = downsize(frame, 30)
        ballCopy = frame.copy() # copy frame
        rimCopy = frame.copy()

        (frame, position, radius) = ballSegmentation(ballCopy, frame, hsvRange, fgbg, frameCount)
        (frame, rimPosition) = rimSegmentation(rimCopy, frame, hsvRange, frameCount)

        if len(position) > 0:
            frame = writePosition(frame, position)
            state = setState(state, radius, position, dt)
            if len(state['velocity']) > 1 and state['velocity'][-1][1] > 0 and trajFlag == False:
                trajectory = getTrajectory(state, rimPosition, dt, trajectory)
                trajFlag = True
            if True:
                frame = showTrajectory(trajectory, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        cv2.imshow("frame", frame)
        frameCount += 1

    # close all open windows
    cap.release()
    cv2.destroyAllWindows()

def getHSV(file):
    hsv = {}
    hsvRange = {}
    with open(file, "r") as read_file:
        hsv = json.load(read_file)
    for item in hsv: # ball and backboard
        hsvRange[item] = {'min': [], 'max': []}
        for color in hsv[item]: # h, s, and v
            hsvRange[item]['min'].append(min(hsv[item][color]))
            hsvRange[item]['max'].append(max(hsv[item][color]))
        hsvRange[item]['min'] = tuple(hsvRange[item]['min'])
        hsvRange[item]['max'] = tuple(hsvRange[item]['max'])
    return hsvRange

def downsize(frame, scale):
    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame

def ballSegmentation(ballCopy, frame, hsvRange, fgbg, frameCount):
    blurred = cv2.bilateralFilter(ballCopy, 9, 75, 75)
    # background subtraction
    fgmask = fgbg.apply(ballCopy, None, 0.01)
    ballCopy = cv2.bitwise_and(frame, frame, mask=fgmask)
    # color thresholding
    ballCopy = cv2.cvtColor(ballCopy, cv2.COLOR_BGR2HSV)
    ballCopy = cv2.inRange(ballCopy, hsvRange['ball']['min'], hsvRange['ball']['max'])
    # denoising
    ballCopy = cv2.erode(ballCopy, None)
    ballCopy = cv2.dilate(ballCopy, None)
    (frame, position, radius) = getContours(ballCopy, frame)
    return (frame, position, radius)

def getContours(ballCopy, frame):
    global side
    ret,thresh = cv2.threshold(ballCopy,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    position = ()
    radius = 0
    props = []
    found = False
    if len(contours) > 0:
        for cnt in contours:
            (x, y), rad = cv2.minEnclosingCircle(cnt)
            props.append((int(rad),int(x),int(y),cnt))
        if len(props) > 0:
            props.sort(key=lambda x: x[0], reverse=True)
            for prop in props:
                r = prop[0]
                x = prop[1]
                y = prop[2]
                c = prop[3]
                if side == 'l':
                    if y < 270 and not x > 410 and r >= 6 and r <= 14: #x >= 165 and y < 200 and r >= 9 and r <= 14:
                        found = True
                        position = (x, y)
                        radius = r
                        break
                else:
                    if y < 270 and x > 410 and r >= 6 and r <= 14: #x >= 165 and y < 200 and r >= 9 and r <= 14:
                        found = True
                        position = (x, y)
                        radius = r
                        break
        if found:
            frame = cv2.circle(frame, position, int(radius), (0,255,0),2)
            frame = cv2.circle(frame, position, 5, (0, 0, 255), -1)
    return (frame, position, radius)

def writePosition(frame, position):
    cv2.putText(frame,'x: {}, y: {}'.format(position[0],position[1]),(position[0]-30,position[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.80,(255,255,255),2,cv2.LINE_AA)
    return frame

def rimSegmentation(rimCopy, frame, hsvRange, count):
    rimPosition = {
        'center': (),
        'leftExtreme': (),
        'rightExtreme': ()
    }
    thr = cv2.bilateralFilter(rimCopy, 7, 75, 75)
    thr = cv2.cvtColor(thr, cv2.COLOR_BGR2HSV)
    thr = cv2.inRange(thr, hsvRange['backboard']['min'], hsvRange['backboard']['max'])
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if y < 248 and y > 150 and x > 380 and x < 450 and w > 20 and w < 30 and h > 15:#x <= 200 and y > 100 and y < 200 and w < 80 and w > 15 and h > 15 and h < 50:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            center = (((w / 2) + x), (y + h))
            mm2Pixel = (squareWidth) / w # mm / pixel
            rimRadius = (rimWidth / mm2Pixel) * (0.5) # width of rim in pixels
            leftExtreme = (int(center[0] - rimRadius), int(center[1]))
            rightExtreme = (int(center[0] + rimRadius), int(center[1]))
            rimPosition['center'] = leftExtreme
            rimPosition['leftExtreme'] = leftExtreme
            rimPosition['rightExtreme'] = rightExtreme
    return (frame, rimPosition)

def sortList(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs, reverse=True)]
    return z

def setState(state, radius, position, dt):
    z = ballWidth / (2 * radius) # distance from camera
    state['position'].append((position[0], position[1], z))
    state = calculateVelocity(state, dt)
    return state

def calculateVelocity(state, dt):
    if len(state['position']) > 1:
        xVel = (state['position'][-1][0] - state['position'][-2][0]) / dt
        yVel = (state['position'][-1][1] - state['position'][-2][1])  / dt
        zVel = (state['position'][-1][2] - state['position'][-2][2]) / dt
        state['velocity'].append((xVel, yVel, zVel))
    return state

def getTrajectory(state, rim, dt, trajectory):
    global yA
    t = len(state['position']) * dt
    # get positions and velocities
    (x,y,z) = (state['position'][-1][0], state['position'][-1][1], state['position'][-1][2])
    (xV,yV,zV) =  (state['velocity'][-1][0], state['velocity'][-1][1], state['velocity'][-1][2])
    r = ballWidth / (2 * z)
    dt = dt / 4
    # calculate acceleration
    yi = state['position'][0][1]
    yf = state['position'][-1][1]
    #yy = [y for (x,y,z) in state['position']]
    #yi = max(yy)
    yA = calcAcc(yf, yi, t)
    for i in range(100):
        global score
        score = False
        x = (x + (xV * dt))
        y = (y + (yV * dt) + ((1/2) * (yA) * (dt**2)))
        z = z + (zV * dt)
        yV = yV + (yA * dt)
        xV = xV
        r = ballWidth / (2 * z)
        trajectory['x'].append(x)
        trajectory['y'].append(y)
        trajectory['z'].append(z)
    index = closest(trajectory['y'], rim['center'][1])
    x = trajectory['x'][index]
    y = trajectory['y'][index]
    if x > rim['leftExtreme'][0] and x < rim['rightExtreme'][0]:
        score = True
    else:
        score = False
    return trajectory

def calcAcc(yf, yi, t):
    yDiff = abs(yf - yi)
    yA = 2*(yDiff/(t**2))
    return yA

def closest(lst, K):
    return lst.index(lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))])

def showTrajectory(trajectory, frame):
    color = ()
    w = []
    for i in range(len(trajectory['x'])):
        x = trajectory['x'][i]
        y = trajectory['y'][i]
        z = trajectory['z'][i]
        if score == True:
            color = (0,255,0)
        else:
            color = (0, 0, 255)
        r = (ballWidth / z) * (1/2)
        if r > 0 and r < 20:
            w.append(r)
        frame = cv2.circle(frame, (int(x),int(y)), int(5), color, -1)
    return frame

if __name__ == '__main__':
    main()
