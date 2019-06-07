#Adapted from https://github.com/Ebonclaw/Mouse-Wearable-Tech---RFID-and-Localization-Grid-Computer-Vision-Enhancement
import datetime
import imutils
import time
import argparse
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import RFID_Reader
from MouseTracker import MouseTracker
mouseAreaMin = 3500
mouseAreaMax = 15000 #avoid recognizing thicc mice as multiple mice
#Main Loop
mouseTrackers = list()
bundleTrackers = list()
prevBundledMice = 0
maxMovement = 30
fileName = "test.txt"
trialName = None

# TODO: Find these numbers
readerMap = [
    (67, 115), (160, 113), (268, 113), (375, 118), (480, 127), (570, 136), #1-(1-6) [y-x]
    (67, 210), (160, 213), (265, 213), (370, 220), (475, 225), (570, 227), #2-(1-6) [y-x]
    (66, 310), (164, 310), (263, 315), (370, 320), (470, 315), (563, 320)  #3-(1-5) [y-x]
]


def sortNearestFree(pos):
    """
    Sorts all non-bundled mice by their proximity to the given location.
    """
    remainingMice = list(filter(lambda x: not x.bundled, mouseTrackers))
    return sorted(remainingMice, key= lambda x: x.distanceFromPos(pos))

def sortNearest(pos):
    """
    Sorts all mice by their proximity to the given location.
    """
    return sorted(mouseTrackers, key= lambda x: x.distanceFromPos(pos))

def sortNearestBundles(pos):
    """
    Sorts all bundles by their proximity to the given location.
    """
    return sorted(bundleTrackers, key= lambda x: x["mice"][0].distanceFromPos(pos))

def setup():
    """
    Adds all mice that can be read by the reader to the trackers.
    """
    mice = RFID_Reader.scan()
    print("done scan")
    seenTags = []
    #open(fileName, "w+").close()
    bundleTrackers = []
    for (tag, Position) in mice:
        mouseList = list(filter(lambda x: x.tag() == tag, mouseTrackers))
        if len(mouseList) is 0:
            mouseTrackers.append(MouseTracker(readerMap[Position], tag))
        else:
            mouseList[0].updatePosition(readerMap[Position], False)
        file = open(fileName, 'a')
        log = str(tag) + ';' + str(pos) +';' + "None" + '\n'
        file.write(log)
        file.close()

def process():
    #camera = cv2.VideoCapture(0)
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    time.sleep(0.25)
    firstFrame = cv2.imread("ref.jpg")
    firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    #firstFrame = cv2.GaussianBlur(firstFrame, (21,21), 0)
    bgsub = cv2.bgsegm.createBackgroundSubtractorGMG()
    startUpIterations = 100
    diffFrameCount = 0
    frameCount = 0
    needPulse = False
    for rawFrame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port=True):
        try:
            #Grab the current frame
            # print("got the frame")
            # (grabbed, frame) = camera.read()
            #
            # #If we could not get the frame, then we have reached the end of the stream.
            # if not grabbed:
            #     break;
            #Convert to grayscale, resize, and blur the frame
            #frame = imutils.resize(frame, width = 500)
            frame = rawFrame.array
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21,21), 0)

            if firstFrame is None:
                # TODO: Add separate background image handling
                firstFrame = gray

            #Compute difference between current and first frame, fill in holes, and find contours
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 60, 255, cv2.THRESH_BINARY)[1]
            #thresh = cv2.adaptiveThreshold(frameDelta, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

            #Watershed
            kernel = np.ones((10, 10), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 3)

            #Determining sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            #Determining sure foreground area
            dist_trans = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            sure_fg = cv2.threshold(dist_trans,0.6*dist_trans.max(),255, 0)[1]

            #Unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            #thresh = cv2.dilate(thresh, None, iterations=2)

            # thresh = bgsub.apply(frame, learningRate = 0.2)

            (rawContours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



            processedContours = list()# clear stream to prepare for next frame
            rawFrame.truncate(0)



            """
            Capstone process:
                If area of contour is bigger than min, it must be at least one mouse.
                If it is less than two mice, it is one.
                If it is greater than minimum for two mice and less than minimum for three, it is two mice.
                If it is greater than min. for three, it is three mice.
                Otherwise, it is not a mouse (something smaller has moved: dust, food, etc.)
                If any mice have merged, record this.
                Then, for all contours that are mice, first check for all single mice.
                Single mice are simple. Find their center and store it in the tracker.
                Merged mice: very complex.
            How can we improve this?
                Idea #1: Simply, don't handle the merge case. At the point the mice merge,
                we treat them as one "bundle" of mice. *If* we can assume the number of mice in the cage is
                constant, then this is simple. For situations such as AHF, we can potentially designate
                a region of the image as an "entrance/exit zone", where we can decrement and increment a
                global mouse counter.
                With this, we may not have to have set sizes for the individual sets of mice,
                which strikes me as a poor idea regardless.
                A maximum size for a mouse should suffice.
                Then, whenever the contour count decrements, we check the distance between
                the bundle and the last known position of the vanished mouse. If it is close enough,
                assume the mouse has joined the bundle. Otherwise, assume the mouse has left the cage.
                Whenever a mouse leaves the bundle (i.e. a new contour appears), verify which it is
                with the RFID system.
                Possible problems:
                    - A mouse could leave the cage at the same time as another leaves the bundle.
                      This system could potentially not notice this.
                    - The size of mouse bundles could become too large to get meaningful data out of.
                    - Multiple bundles forming nearby each other?
            """
            bundleCount = 0
            #If any error occurs, scan the entire base and update mouse positions to RFID tags
            error = False
            updated = False
            for contour in rawContours:
                if cv2.contourArea(contour) < mouseAreaMin:
                    #Not a mouse :(
                    continue
                elif cv2.contourArea(contour) < mouseAreaMax:
                    #This is just one mouse
                    moments = cv2.moments(contour)
                    centerX = int(moments["m10"] / moments["m00"])
                    centerY = int(moments["m01"] / moments["m00"])
                    processedContours.append({'contour': contour, 'bundle': False, 'center': (centerX, centerY)})
                    rotated_box = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rotated_box)
                    box = np.int0(box)
                    #Green Box
                    cv2.drawContours(frame, [box], 0, (0, 255, 0),2)
                else:
                    #This is multiple mice
                    moments = cv2.moments(contour)
                    centerX = int(moments["m10"] / moments["m00"])
                    centerY = int(moments["m01"] / moments["m00"])
                    processedContours.append({'contour': contour, 'bundle': True, 'center': (centerX, centerY)})
                    rotated_box = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rotated_box)
                    box = np.int0(box)
                    #Red Box
                    cv2.drawContours(frame, [box], 0, (0, 0, 255),2)


            #processedContours = []

            prevFreeMice = list(filter(lambda x: not x.bundled, mouseTrackers))
            freeMouseContours = list(filter(lambda x: not x["bundle"], processedContours))
            bundleContours = list(filter(lambda x: x['bundle'], processedContours))
            if len(freeMouseContours) ==len(prevFreeMice) and needPulse:
                diffFrameCount += 1
                if diffFrameCount < 15:
                    #Give enough time for mice to be clearly separated
                    continue
                #A good base. Pulse nearby RFIDs to determine mouse positions.
                for contour in freeMouseContours:
                    #Update mouse with tag
                    x2 = contour["center"][0]
                    y2 = contour["center"][1]
                    nearestReaders = sorted(readerMap, key= lambda x: np.sqrt(((x[0]-x2)*(x[0]-x2) + (x[1]-y2)*(x[1]-y2))))
                    #print(nearestTagIndex)
                    #tag readers are upside down, fix later
                    tag = False
                    index = 0
                    index = readerMap.index(nearestReaders[0])
                    tag = RFID_Reader.readTag(17 - index)
                    if tag is False:
                        #Try again next time
                        error = True
                        break
                    tag = tag[0]
                    mouseList = list(filter(lambda x: x.tag() == tag, mouseTrackers))
                    if len(mouseList) is 0:
                        print("brand new")
                        print(list(map(lambda x: x.tag(), mouseTrackers)))
                        mouseTrackers.append(MouseTracker(readerMap[index], tag))
                        mouseList = list(filter(lambda x: x.tag() == tag, mouseTrackers))
                        mouseList[0].updatePosition(readerMap[index], False)
                    else:
                        mouseList[0].updatePosition(readerMap[index], False)

                for proContour in freeMouseContours:
                    mouse = sortNearestFree(proContour["center"])[0]
                    mouse.updatePosition(proContour["center"], False)
                needPulse = False
                updated = True

            elif len(freeMouseContours) < len(prevFreeMice):
                if len(bundleContours) == 0:
                    diffFrameCount = 0
                    #Mice have climbed on top of each other(probably)
                    needPulse = True
                # diffFrameCount += 1
                # print("bundle")
                # if diffFrameCount <= 5:
                #     #Ignore frames of mice briefly passing by each other.
                #     #Slows the algorithm significantly to process these.
                #     continue
                # #Some mice have joined new bundles.
                # #For free mice, simple. Update all the remaining free mice.
                # remainingMice = mouseTrackers.copy()
                # for proContour in freeMouseContours:
                #     try:
                #         mouse = sortNearestFree(proContour["center"])[0]
                #         mouse.updatePosition(proContour["center"], False)
                #         remainingMice.remove(mouse)
                #     except Exception as e:
                #         error = True
                # #Bundles: Form new bundles or make bigger ones
                # for proContour in bundleContours:
                #     nearestMice = sortNearest(proContour["center"])
                #     if(nearestMice[0].bundled and len(bundleTrackers) >0):
                #         #This is a previously created bundle! (Mice in a bundle have same position as bundle center)
                #         print(bundleTrackers)
                #         try:
                #             bundle = sortNearestBundles(proContour["center"])[0]
                #             bundle["position"] = proContour["center"]
                #             for mouse in bundle["mice"]:
                #                 mouse.updatePosition(proContour["center"], True)
                #         except Exception as e:
                #             error = True
                #         continue
                #     else:
                #         #New bundle!
                #         #First two will *always* be part of the bundle, otherwise the bundle would be merged with another.
                #         mice = []
                #         print(len(remainingMice))
                #         if len(remainingMice) < 2:
                #             error = True
                #             break
                #         mice.append(nearestMice[0])
                #         #This mouse *has* to be in remaining mice, otherwise it is both the closest
                #         #to a free contour and a bundle contour, which is impossible.
                #         try:
                #             remainingMice.remove(nearestMice[0])
                #             nearestMice[0].updatePosition(proContour["center"], True)
                #             mice.append(nearestMice[1])
                #             nearestMice[1].updatePosition(proContour["center"], True)
                #             remainingMice.remove(nearestMice[1])
                #             bundleTrackers.append({"position": proContour["center"], "mice": mice, "processed": False})
                #         except Exception as e:
                #             error = True
                # #Now any remaining mice must be in a bundle.
                # for mouse in remainingMice:
                #     try:
                #         if len(bundleContours) is 0 or len(bundleTrackers) is 0:
                #            #Mouse has left
                #             print("mouse left")
                #             mouseTrackers.remove(mouse)
                #             continue
                #         nearestBundle = min(bundleContours, key=lambda x: mouse.distanceFromPos(x["center"]))
                #         if mouse.distanceFromPos(nearestBundle["center"]) > maxMovement:
                #             #Mouse has left (or we lost it)
                #             print("mouse left")
                #             mouseTrackers.remove(mouse)
                #             continue
                #         mouse.updatePosition(nearestBundle["center"], True)
                #         bundle = sortNearestBundles(proContour["center"])[0]
                #         bundle["mice"].append(mouse)
                #     except Exception as e:
                #         error = true
            elif len(freeMouseContours) > len(prevFreeMice):
                pass
                # diffFrameCount += 1
                # if diffFrameCount <= 5:
                #     continue
                # print("separate")
                # #Some mice have left their bundles, or new mice have arrived.
                # for mouse in prevFreeMice:
                #     try:
                #         nearestContour = sorted(freeMouseContours, key=lambda x: mouse.distanceFromPos(x["center"]))[0]
                #         mouse.updatePosition(nearestContour["center"], False)
                #         processedContours.remove(nearestContour)
                #         freeMouseContours.remove(nearestContour)
                #     except Exception as e:
                #         error = True
                # for contour in freeMouseContours:
                #     #Update mouse with tag
                #     x2 = contour["center"][0]
                #     y2 = contour["center"][1]
                #     nearestReaders = sorted(readerMap, key= lambda x: np.sqrt(((x[0]-x2)*(x[0]-x2) + (x[1]-y2)*(x[1]-y2))))
                #     #print(nearestTagIndex)
                #     #tag readers are upside down, fix later
                #     tag = False
                #     index = 0
                #     num = 0
                #     count = 0
                #     while tag is False:
                #         index = readerMap.index(nearestReaders[num])
                #         tag = RFID_Reader.readTag(17 - index)
                #         count+= 1
                #         if count > 3:
                #             break
                #     if tag is False:
                #         #Try again next time
                #         error = True
                #         break
                #     tag = tag[0]
                #     mouseList = list(filter(lambda x: x.tag() == tag, mouseTrackers))
                #     if len(mouseList) is 0:
                #         print("brand new")
                #         print(list(map(lambda x: x.tag(), mouseTrackers)))
                #         mouseTrackers.append(MouseTracker(readerMap[index], tag))
                #         mouseList = list(filter(lambda x: x.tag() == tag, mouseTrackers))
                #         mouseList[0].updatePosition(readerMap[index], False)
                #     else:
                #         mouseList[0].updatePosition(readerMap[index], False)
                #     #Update remaining bundles
                #     for proContour in bundleContours:
                #         bundle = sortNearestBundles(proContour["center"])[0]
                #         bundle["position"] = proContour["center"]
                #         bundle["processed"] = True
                #         for mouse in bundle["mice"]:
                #             if mouse.bundled:
                #                 mouse.updatePosition(proContour["center"], True)
                #             else:
                #                 bundle["mice"].remove(mouse)
                #     #Remove any unprocessed bundles (these are now empty)
                #     for bundle in bundleTrackers:
                #         if bundle["processed"]:
                #                 bundle["processed"] = False
                #         else:
                #             bundleTrackers.remove(bundle)
            if error:
                #Not a good set of tags
                continue
                #Refresh from RFID
                #setup()
            frameName = "tracking_system:" + trialName + str(frameCount) + ".png"
            frameCount += 1
            if updated:
                for mouse in mouseTrackers:
                    pos = mouse.getPosition()
                    cv2.putText(frame, str(mouse.tag()), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    file = open(fileName, 'a')
                    log = str(mouse.tag()) + ';' + str(pos) +';' + frameName + '\n'
                    file.write(log)
                    file.close()
            cv2.imshow("Mouse Tracking", frame)
            key = cv2.waitKey(1)& 0xFF
            cv2.imwrite("FrameData/" + frameName, frame)


            if key==ord('q'):
                break
        except KeyboardInterrupt:
            break





if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--text", help="path to the text file")
    ap.add_argument("-n", "--name", default ="base_tracking", help="trial name")
    args = vars(ap.parse_args())

    if args.get("text", None) is not None:
        fileName = args.get('text')
        open(fileName, "w+").close()
    trialName = args.get("name")
    print('hello')
    #setup()
    process()

        # for proContour in list(filter(lambda x: x['bundle'], processedContours)):
        #     #First two will *always* be part of the bundle, otherwise the bundle would be merged with another.
        #     mice = []
        #     mice.append(sortNearestFree(proContour.center)[0])
        #     sortNearestFree(proContour.center)[0].updatePosition(proContour.center, True)
        #     mice.append(sortNearestFree(proContour.center)[1])
        #     sortNearestFree(proContour.center)[1].updatePosition(proContour.center, True)
        #     bundleTrackers.append({"position": proContour.center, "mice": mice})
        #     miceToBundle -=2
