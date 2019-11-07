"""
Final processing file. Copy into the darknet folder to run.
Adapted from https://github.com/AlexeyAB/darknet
"""

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
import itertools
from munkres import Munkres
from shutil import copyfile
from MouseTracker import MouseTracker

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, mice):
    for mouse in mice:
        if mouse.visualTracker is not None:
            pos = mouse.getPosition()
            x, y, w, h =  pos[0], pos[1], pos[4], pos[5]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (255, 0, 255), 1)
            cv2.putText(img,
                 str(mouse.tag()) +
                 " [vis]",
                  (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  [255, 0, 255], 2)

    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        vx = detection[3][0]
        vy = detection[3][1]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.rectangle(img, (entranceX, entranceY), (640, 640), [0, 120,120])
        cv2.circle(img, (int(x), int(y)), 5, [0, 0, 255])
        cv2.circle(img, (int(550*640/640), int(350*640/480)), 5, [0, 255, 0])
        cv2.circle(img, (int(550*640/640), int(100*640/480)), 5, [0, 255, 0])
        cv2.circle(img, (int(100*640/640), int(100*640/480)), 5, [0, 255, 0])
        cv2.circle(img, (int(100*640/640), int(350*640/480)), 5, [0, 255, 0])
        cv2.circle(img, (int(x), int(y)), maxSwapDistance, [255, 0, 0])
        cv2.arrowedLine(img, (int(x - vx/2), int(y - vy/2)), (int(x + vx/2), int(y + vy/2)), [0, 0, 255])
        cv2.putText(img,
                    str(detection[0]) +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)

    return img


netMain = None
metaMain = None
altNames = None
entranceX = 550
entranceY = 350
maxSwapDistance = 100
minSwapIOU = 0.2


def miceWithinDistance(mice, distance):
    within_dist = False
    for mouse in mice:
        for other in mice:
            if other is not mouse and mouse.distanceFromPos(other.getPosition()) < distance:
                within_dist = True
    return within_dist


def YOLO(trialName, mice, RFID, showVideo):
    miceNum = len(mice)
    global dataPath, dataDrive, useFrames
    global metaMain, netMain, altNames
    configPath = "./yolo-obj.cfg"
    weightPath = "./yolo-obj_new.weights"
    metaPath = "./data/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    if not useFrames:
        cap = cv2.VideoCapture(dataDrive + dataPath + "/tracking_system" + trialName + ".h264")
    frameCount = 1
    """
    Problem: Mice disappear and then reappear while others are unidentified. They should be instantly identified but are not.
    Essentially, need secondary handling before moving into "officially lost"
    So, we need "partially lost" and "officially lost".
    Partially lost mice can be found again as soon as the detection reappears.
    Officially lost mice need the RFID tags to verify them.
    """
    lostTrackers = []
    partialLostTrackers = []
    error = False
    dummyTag = 0
    badFrameCount = 0
    lastRFIDIndex = -1
    pairFinder = Munkres()
    event = {"identity_swap": [], 'dummy_swap': [], 'lost_mouse': [], 'visual' : []} #dictionary to hold events key = event name, value = number of occurences
    badDetections = 0
    for mouse in mice:
        lostTrackers.append(mouse)
        mice = []

    while True:
        RFIDIndices = []
        try:
            validationFrame = False
            frameName = dataDrive + dataPath + "/tracking_system" + trialName + str(frameCount) + ".jpg"
            while dataDrive + dataPath + "/" + RFID[lastRFIDIndex + 1].split(';')[2].strip('\n') == frameName:
                RFIDIndices.append(lastRFIDIndex + 1)
                validationFrame = True
                lastRFIDIndex += 1
            if useFrames:
                frame_read = cv2.imread(frameName)
            else:
                success, frame_read = cap.read()
                if not success:
                    break
            print(frameCount)
            frameCount += 1
            frame_rgb = frame_read
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(str(e))
            # no more frames
            break
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)

        updatedTags = []
        # if len(detections) > miceNum:
        #     # Sort by the likelihood that it is a mouse, remove the least likely ones
        #     sortedDetections = sorted(detections, key=lambda l: l[1])
        #     while len(sortedDetections) > miceNum:
        #         sortedDetections.remove(sortedDetections[0])
        #     detections = sortedDetections

        cleanedDetections = []

        """
        ========================================Track Assignment=======================================================
        Approach 1: Assign each detection to its nearest mouse.
        Problem: When a mouse reappears, the detection is closer to a mouse that has been tracked throughout than the mouse.
        Approach 2: Assign each mouse to its nearest detection.
        Problem: When a mouse disappears, it can be assigned to a detection even if it was supposed to be gone.
        Approach 3: Sort pairs of detection and mouse by distance. Assign in order, until we run out of detections or mice.
        """
        # pairs = itertools.product(list(filter(lambda x: x.lastFrameCount == frameCount -1, mice)), detections)
        pairs = itertools.product(mice, detections)
        pairs = sorted(pairs, key=lambda l: l[0].trackLikelihood(l[1][2]), reverse = True)
        matrix = []
        for i in range(0, len(mice)):
            row = []
            for x in range (0, len(mice)):
                if x >= len(detections):
                    row.append(0)
                else:
                    d = detections[x]
                    row.append(mice[i].distanceFromPos((d[2][0], d[2][1])))
            matrix.append(row)
        # pairIndices = pairFinder.compute(matrix)
        """
        ################ MAIN DETECTION ASSIGNMENT #######################
        """
        for mouse, detection in pairs:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            if mouse.tag() not in updatedTags and detection in detections:
                updatedTags.append(mouse.tag())
                if len(partialLostTrackers) == 1 and mouse.tag() == partialLostTrackers[0].tag():
                    if mouse.intersectionOverUnion(detection[2]) == 0 and x > entranceX and y > entranceY:
                        #Likely a new mouse rather than an old one
                        continue
                    partialLostTrackers = []
                if mouse.visualTracker is not None:
                    mouse.stopVisualTracking(delete=False)
                mouse.updatePosition([x, y], frameName, frameCount -1, w, h)
                updatedDetection = list(detection)
                updatedDetection[0] = mouse.tag()
                updatedDetection.append(mouse.velocity)
                detections.remove(detection)
                cleanedDetections.append(updatedDetection)
                if len(partialLostTrackers) == 1 and mouse.tag() == partialLostTrackers[0].tag():
                    partialLostTrackers = []
        """
        Any unassigned detections are given dummy tags.
        """
        sortedDetections = sorted(detections, key=lambda l: l[1])
        for detection in sortedDetections:
            if len(mice) >= miceNum:
                break
            updatedTags.append(dummyTag)
            mice.append(MouseTracker([detection[2][0], detection[2][1]], dummyTag, frameName, frameCount -1, detection[2][2], detection[2][3]))
            dummyTag += 1

        image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if showVideo:
            cvDrawBoxes(cleanedDetections, image, mice)
            cv2.imshow('Demo', image)
            cv2.waitKey(3)

        # =============================Lost Mouse Handling=====================================================
        if len(cleanedDetections) < miceNum:
            badDetections += 1
            #  Mice have been lost this frame!
            error = True
            # ========================Visual Tracking============================================
            for tracker in filter(lambda x: x.tag() not in updatedTags, mice):
                if tracker.visualTracker != None:
                    ok, bbox = tracker.visualTracker.update(frame_resized)
                    if ok:
                        print('vis')
                        event['visual'].append((frameName, tracker.tag()))
                        tracker.updatePosition([bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2], frameName, frameCount, bbox[2], bbox[3])
                    else:
                        tracker.stopVisualTracking()
            #====================Adding to Partial Lost================================
            print(updatedTags)
            print(list(map(lambda x: x.tag(), partialLostTrackers)), 'PL')
            for tracker in filter(lambda x: x.tag() not in updatedTags and x not in partialLostTrackers and x.visualTracker == None, mice):
                print(tracker.tag(), tracker.getPosition(), 'partial lost')
                if tracker.canDoVisual:
                    tracker.startVisualTracking(frame_resized)
                numClose = len(partialLostTrackers)
                nearest = sorted(list(filter(lambda x: x.tag() in updatedTags, mice)), key= lambda l: l.distanceFromPos(tracker.getPosition()))
                for nMouse in nearest:
                    if nMouse not in partialLostTrackers:
                        # This checks if there is a mouse nearby to the one that just disappeared.
                        iou = nMouse.trackLikelihood([tracker.getPosition()[0], tracker.getPosition()[1], tracker.getPosition()[4], tracker.getPosition()[5]])
                        print(nMouse.tag(), iou)
                        if iou <= minSwapIOU:
                            print("no intersection")
                            pass
                        else:
                            partialLostTrackers.append(nMouse)
                #If no mice are near and was in entrance area, then do not add to partial lost, they likely left the cage.
                numClose = len(partialLostTrackers) - numClose
                print(numClose, 'nearbyMice')
                if numClose == 0 and tracker.getPosition()[0] > entranceX and tracker.getPosition()[1] > entranceY:
                     event["lost_mouse"].append((frameName, tracker.tag()))
                     print("mouse exited")
                     if tracker.tag() > 99999:
                         # This is not a dummy tracker, we can find it again
                         lostTrackers.append(tracker)
                     try:
                         mice.remove(tracker)
                     except Exception as e:
                         print(str(e))
                         # for safety idk
                         pass
                elif len(tracker.recordedPositions) <= 5 and tracker.tag() < 99999:
                    mice.remove(tracker)
                else:
                    partialLostTrackers.append(tracker)

        # ================ Partial Lost Emptying =====================
        if len(partialLostTrackers) > 1:
            # Lost more than one? Then once it reappears we cannot know which it is.
            # We now require the RFID.

            for tracker in partialLostTrackers:
                #print(track.tag(), "partial lost")
                event["lost_mouse"].append((frameName, tracker.tag()))
                if tracker.tag() > 99999:
                    # This is not a dummy tracker, we can find it again
                    lostTrackers.append(tracker)
                try:
                    mice.remove(tracker)
                except ValueError:
                    # for safety idk
                    pass
            partialLostTrackers = []

        # ======================================RFID Validation========================================================
            # Check if we can match up a dummy mouse with a tag
        if validationFrame:
            for RFIDIndex in RFIDIndices:
                ln = RFID[RFIDIndex].split(';')
                readerPos = list(int(item) for item in ln[1].strip('()\n').split(','))
                usedIndex = False
                """
                First iterate through all lost trackers. If the pickup is one of
                these, we either have an identification or a dummy swap.
                """
                for tracker in partialLostTrackers:
                    if int(ln[0]) == tracker.tag():
                        #You have just got a pickup for a mouse that has recently disappeared.
                        #In all likelihood, they are occluded by another, using
                        #this pickup will do nothing but harm,
                        #Ignore this pickup.
                        usedIndex = True
                for tracker in lostTrackers:
                    if int(ln[0]) == tracker.tag():
                        usedIndex = True
                        # Match!
                        readerPos[0] *= 640/640
                        readerPos[1] *= 640/480
                        nearestMice = sorted(mice, key= lambda x: x.distanceFromPos(readerPos))
                        if len(nearestMice) < 1 or nearestMice[0].distanceFromPos(readerPos) > 300:
                            #If nearest mouse is not currently detected. do nothing
                            break
                        if len(nearestMice) > 2 and abs(nearestMice[0].distanceFromPos(readerPos) - nearestMice[1].distanceFromPos(readerPos)) < 75:
                            # We cannot be certain which one is over the reader
#                                print("cannot be sure")
                            break
                        # Update Dummy Track
                        if nearestMice[0].tag() < 99999:
                            print("found mouse again")
                            nearestAnon = nearestMice[0]
                            tracker.updatePositions(nearestAnon.recordedPositions)
                            tracker.validate()
                            mice.append(tracker)
                            lostTrackers.remove(tracker)
                            mice.remove(nearestAnon)
                            #DUMMY SWAP
                        else:
                            # There was an identity swap earlier. Correct for it AND increment number of identity swap
                            print("dummy swap")
                            event["dummy_swap"].append((frameName, nearestMice[0].tag()))
                            badMouse = nearestMice[0]
                            occlusionEndPoint = badMouse.occlusionPointBefore(list(filter(lambda x: x.tag() != badMouse.tag(), mice)), maxSwapDistance)
                            incorrectPositions = badMouse.trimPositions(occlusionEndPoint)
                            print("occlusion End", occlusionEndPoint)
                            tracker.updatePositions(incorrectPositions)
                            tracker.validate()
                            occlusionStartPoint = badMouse.occlusionPointAfter(list(filter(lambda x: x.tag() != badMouse.tag(), mice)), maxSwapDistance)
                            print("occlusion Start", occlusionStartPoint)
                            badMouse.trimPositions(occlusionStartPoint)
                            mice.append(tracker)
                            lostTrackers.remove(tracker)
                            lostTrackers.append(badMouse)
                            mice.remove(badMouse)
                if not usedIndex:
                    """
                    In this case, the pickup was not one of the lost trackers.
                    Therefore, we either have a validation of an existing track
                    or an identity swap.
                    """
                    nearestMice = sorted(mice, key= lambda x: x.distanceFromPos(readerPos))
                    actualMouse = list(filter(lambda x: x.tag() == int(ln[0]), mice))[0]
                    if len(nearestMice) < 1 or nearestMice[0].distanceFromPos(readerPos) > 300:
                        #If nearest mouse is not currently detected. do nothing
                        break
                    if len(nearestMice) < 2 or abs(nearestMice[0].distanceFromPos(readerPos) - nearestMice[1].distanceFromPos(readerPos)) > 75:
                        #IDENTITY SWAP
                        if nearestMice[0].tag() != int(ln[0]):
                            #Identity swap, increment number identity swaps
                            # An identity swap has occured. Remove the frames up to the last validation point.
                            #print(list(map(lambda x:x.tag(), mice)))
                            print("identity swap")

                            event["identity_swap"].append((frameName, [nearestMice[0].tag(), actualMouse.tag()]))
                            #Find first valid point of current track
                            actualOcclusionEndPoint = actualMouse.occlusionPointBefore(list(filter(lambda x: x.tag() != actualMouse.tag(), mice)), maxSwapDistance)
                            nearOcclusionEndPoint = nearestMice[0].occlusionPointBefore(list(filter(lambda x: x.tag() != nearestMice[0].tag(), mice)), maxSwapDistance)
                            nearPositions_new = actualMouse.trimPositions(actualOcclusionEndPoint)
                            actualPositions_new = nearestMice[0].trimPositions(nearOcclusionEndPoint)
                            #Find last valid point of old track
                            nearOcclusionStartPoint = actualMouse.occlusionPointAfter(list(filter(lambda x: x.tag() != actualMouse.tag(), mice)), maxSwapDistance)
                            actualOcclusionStartPoint = nearestMice[0].occlusionPointAfter(list(filter(lambda x: x.tag() != nearestMice[0].tag(), mice)), maxSwapDistance)
                            #remove all occlusioned track
                            actualMouse.trimPositions(actualOcclusionStartPoint)
                            nearestMice[0].trimPositions(nearOcclusionStartPoint)
                            #Reassign current tracks
                            actualMouse.updatePositions(actualPositions_new)
                            actualMouse.validate()
                            nearestMice[0].updatePositions(nearPositions_new)
                        #VALIDATION
                        else:
                            nearestMice[0].validate()
            if len(lostTrackers) == 1:
                # Only one lost mouse = only one possibility
                if len(list(filter(lambda x: x.tag() == lostTrackers[0].tag(), mice))) >= 1:
                    lostTrackers = []
                    error = False
                else:
 #                   print("Only one unknown, assigned")
  #                  print(list(map(lambda x: x.tag(), mice)), "mice")
   #                 print(list(map(lambda x: x.tag(), lostTrackers)), "lost")
                    missingMouse = list(filter(lambda x: x.tag() < 99999, mice))
                    if len(missingMouse) > 0:
                        missingMouse = missingMouse[0]
                        lostTrackers[0].updatePositions(missingMouse.recordedPositions)
                        lostTrackers[0].validate()
                        mice.append(lostTrackers[0])
                        lostTrackers = []
                        mice.remove(missingMouse)
                        error = False
            if len(lostTrackers) == 0:
                error = False

        print(list(map(lambda x: x.tag(), mice)),"mice")
        print(list(map(lambda x: x.tag(), lostTrackers)), "lost")
#        input("next")
#        time.sleep(0.2)
    mouseDict = {}
    for mouse in mouseTrackers:
        mouseDict.update({mouse.tag(): mouse.recordedPositions})
        print(mouse.tag(), str(len(mouse.recordedPositions)/frameCount*100) + "% Covered")
    print(list(map(lambda x: str(x[0]) + ":" + str(len(x[1])), event.items())))
    print("frames without all mice: ", badDetections, badDetections/frameCount*100,  "% error")
    with open(dataDrive + dataPath + "/processed.json", "w") as outfile:
        json.dump(mouseDict, outfile, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="1 for showing video, 0 or blank for not")
    ap.add_argument("-f", "--frames", help="Include this argument if you have individual frame files")
    ap.add_argument("-d", "--drive", help="Path to data")
    ap.add_argument("-n", "--name", help="Name of the frame folder/text file")
    args = vars(ap.parse_args())
    showVideo = False
    useFrames = False
    if args.get("video", None) is not None:
        showVideo = True
    if args.get("frames", None) is not None:
        useFrames = True
    mouseTrackers = []
    tags = tuple(int(x.strip()) for x in input("Please input the mouse tags, separated by commas").split(','))
    print(tags)
    for tag in tags:
        mouseTrackers.append(MouseTracker([0, 0], tag))
    RFIDResponses = []
    dataPath = args.get("name", "")
    dataDrive = args.get("drive", "frameData")
    if dataDrive  == "frameData":
        dataFileName = "RTS_test"
        dataFileName += args.get("name", "") + ".txt"
    else:
        dataFileName = dataDrive + dataPath + "/RTS_test.txt"
    file = open(dataFileName, "r")
    RFIDResponses = file.readlines()
    YOLO("base_tracking", mouseTrackers, RFIDResponses, showVideo)
