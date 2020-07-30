"""
Final processing file. Copy into the darknet folder to run.
Adapted from https://github.com/AlexeyAB/darknet
"""

from ctypes import *
import math
import random
import traceback
import sys
import os
import cv2
import numpy as np
from copy import copy, deepcopy
import time
from naturalmousetracker.detection_utils import darknet
import json
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
from naturalmousetracker import data
from naturalmousetracker.detection_utils.ImageProcessing import ImageProcessing
import argparse
import itertools
from munkres import Munkres
from tqdm import tqdm
from shutil import copyfile
from naturalmousetracker.detection_utils.readEncoding import decode
from naturalmousetracker.detection_utils.MouseTracker import MouseTracker

netMain = None
metaMain = None
altNames = ["mouse"]
readerMap = [[100, 467], [550, 467], [100, 133], [550, 133]]
entranceX = 580
entranceY = 250
maxSwapDistance = 100
minSwapIOU = 0.2


def YOLO(trialName, trackedMice, RFID, showVideo):
    miceNum = len(trackedMice)
    global dataPath, dataDrive, useFrames, verbose, tags
    global metaMain, netMain, altNames
    #Loading required darknet files
    with pkg_resources.path(data, 'yolo-obj.cfg') as configPath, pkg_resources.path(data, 'yolo-obj_best.weights') as weightPath, pkg_resources.path(data, 'obj.data') as metaPath:
        if not os.path.exists(configPath.absolute()):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath.absolute())+"`")
        if not os.path.exists(weightPath.absolute()):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath.absolute())+"`")
        if not os.path.exists(metaPath.absolute()):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath.absolute())+"`")
        configPath = str(configPath.absolute())
        weightPath = str(weightPath.absolute())
        metaPath = str(metaPath.absolute())
        if netMain is None:
            netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        print("loading meta")
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
        printCheck("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    if not useFrames:
        cap = cv2.VideoCapture(dataDrive + dataPath + "/tracking.h264")

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
    for mouse in trackedMice:
        lostTrackers.append(mouse)
        trackedMice = []
    table = np.array([((i/255.0) ** 0.8)*255 #0.7 to 0.9 seems like a good range.
        for i in np.arange(0, 256)]).astype('uint8')
    pbar = tqdm(total=18000)
    outMatrix = []
    while True:
        RFIDIndices = []
        outRow = {"Frame": 0, "Raw Detections": [], "RFID Data": "", "Tracked Mice": [], "Lost Tracks": [], "Partially Lost Tracks": [], "Events": ""}
        try:
            validationFrame = False
            frameName = dataDrive + dataPath + "/tracking_system" + trialName + str(frameCount) + ".jpg"
            # while dataDrive + dataPath + "/" + RFID[lastRFIDIndex + 1].split(';')[2].strip('\n') == frameName:
            #     RFIDIndices.append(lastRFIDIndex + 1)
            #     validationFrame = True
            #     lastRFIDIndex += 1

            if useFrames:
                frame_read = cv2.imread(frameName)
            else:
                success, frame_read = cap.read()
                if not success:
                    print("Had problems with frame_read")
                    break
            if decode(frame_read) != -1:
                validationFrame = True
                validationTag, validationReader = decode(frame_read)
                printCheck(decode(frame_read))
                validationTag = tags[validationTag]
                printCheck(validationReader)
                validationReader = readerMap[validationReader]
                outRow["RFID Data"] = str(validationTag) + ": " + str(validationReader)
            printCheck(frameCount)
            frameCount += 1
            outRow["Frame"] = frameCount
            pbar.update()
            frame_rgb = frame_read
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)
            frame_resized = cv2.LUT(frame_resized, table)
        except Exception as e:
            printCheck(str(e))
            # no more frames
            break
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)
        outRow["Raw Detections"] = deepcopy(detections)

        updatedTags = []
        # if len(detections) > miceNum:
        #     # Sort by the likelihood that it is a mouse, remove the least likely ones
        #     sortedDetections = sorted(detections, key=lambda l: l[1])
        #     while len(sortedDetections) > miceNum:
        #         sortedDetections.remove(sortedDetections[0])
        #     detections = sortedDetections

        cleanedDetections = []


        for mouse in partialLostTrackers: #Every frame increment all lostCounters who are not -1
            if mouse.lostCounter >= 30:
                printCheck("Counter of ", mouse.tag(), "above thresh, removing from PL")
                if mouse.id > 99999:
                    lostTrackers.append(mouse)
                    mouse.lostCounter = -1
                if mouse in trackedMice:
                    trackedMice.remove(mouse)
                partialLostTrackers.remove(mouse)
                outRow["Events"] += mouse.tag() + " partial lost expired; "

            if mouse.lostCounter >= 0:
                mouse.lostCounter += 1


        """
        ========================================Track Assignment=======================================================
        Approach 1: Assign each detection to its nearest mouse.
        Problem: When a mouse reappears, the detection is closer to a mouse that has been tracked throughout than the mouse.
        Approach 2: Assign each mouse to its nearest detection.
        Problem: When a mouse disappears, it can be assigned to a detection even if it was supposed to be gone.
        Approach 3: Sort pairs of detection and mouse by distance. Assign in order, until we run out of detections or trackedMice.
        """
        # pairs = itertools.product(list(filter(lambda x: x.lastFrameCount == frameCount -1, trackedMice)), detections)
        pairs = itertools.product(trackedMice, detections)
        pairs = sorted(pairs, key=lambda l: l[0].trackLikelihood(l[1][2], frame_resized), reverse = True)

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
                    if mouse.trackLikelihood(detection[2], frame_resized) < -5:
                        #Likely a new mouse rather than an old one
                        continue
                    partialLostTrackers = []
                if mouse.trackLikelihood(detection[2], frame_resized) < -5:
                    #Definitely not the same detection.
                    continue
                if mouse.visualTracker is not None:
                    mouse.stopVisualTracking(delete=False)
                mouse.updatePosition([x, y], frameName, frameCount -1, w, h)
                updatedDetection = list(detection)
                updatedDetection[0] = mouse.tag()
                updatedDetection.append(mouse.velocity)
                detections.remove(detection)
                cleanedDetections.append(updatedDetection)
                if len(partialLostTrackers) > 0 and mouse.tag() == partialLostTrackers[0].tag():
                    mouse.lostCounter = -1
                    partialLostTrackers.remove(mouse)

        """
        Any unassigned detections are given dummy tags.
        """
        sortedDetections = sorted(detections, key=lambda l: l[1])
        for detection in sortedDetections:
            ignoreDetection = False
            for mouse in filter(lambda x: x.tag() in updatedTags, trackedMice):
                if mouse.intersectionOverUnion(detection[2]) != 0:
                    if detection[2][0] > entranceX and detection[2][1] > entranceY:
                        break
                    ignoreDetection = True
                    for lost in lostTrackers + partialLostTrackers:
                        if len(lost.recordedPositions) > 0 and (frameCount - lost.recordedPositions[len(lost.recordedPositions)-1][3]) < 10:
                            ignoreDetection = False
                        if len(lost.recordedPositions) == 0:
                            ignoreDetection = False

            if ignoreDetection:
                continue
            if len(trackedMice) >= miceNum:
                break
            updatedTags.append(dummyTag)
            newMouse = MouseTracker([detection[2][0], detection[2][1]], dummyTag, frameName, frameCount -1, detection[2][2], detection[2][3])
            newMouse.updateHistogram(frame_resized)
            trackedMice.append(newMouse)
            dummyTag += 1

        image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # =============================Lost Mouse Handling=====================================================
        if len(cleanedDetections) < miceNum:
            badDetections += 1
            #  Mice have been lost this frame!
            error = True
            # ========================Visual Tracking============================================
            for tracker in filter(lambda x: x.tag() not in updatedTags, trackedMice):
                if tracker.canDoVisual:
                    tracker.startVisualTracking(frame_resized)
                    outRow["Events"] += tracker.tag() + " started visual tracking; "
                if tracker.visualTracker != None:
                    ok, bbox = tracker.visualTracker.update(frame_resized)
                    if ok:
                        printCheck('vis')
                        event['visual'].append((frameName, tracker.tag()))
                        tracker.updatePosition([bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2], frameName, frameCount, bbox[2], bbox[3])
                        if tracker.distanceFromPos(tracker.visualStartPoint) > 50:
                            outRow["Events"] += tracker.tag() + " stopped visual tracking; "
                            tracker.stopVisualTracking()
                    else:
                        outRow["Events"] += tracker.tag() + " stopped visual tracking; "
                        tracker.stopVisualTracking()
            #====================Adding to Partial Lost================================

            printCheck(list(map(lambda x: x.tag(), partialLostTrackers)), 'PL')
            for tracker in filter(lambda x: x.tag() not in updatedTags and x not in partialLostTrackers and x.visualTracker == None, trackedMice):

                printCheck(tracker.tag(), tracker.getPosition(), 'partial lost')
                numClose = len(partialLostTrackers)
                nearest = sorted(list(filter(lambda x: x.tag() in updatedTags, trackedMice)), key= lambda l: l.distanceFromPos(tracker.getPosition()))
                for nMouse in nearest:
                    if nMouse not in partialLostTrackers:
                        # This checks if there is a mouse nearby to the one that just disappeared.
                        if len(tracker.getPosition()) < 6:
                            iou = 0
                        else:
                            iou = nMouse.trackLikelihood([tracker.getPosition()[0], tracker.getPosition()[1], tracker.getPosition()[4], tracker.getPosition()[5]], frame_resized)
                        printCheck(nMouse.tag(), iou)
                        if iou <= minSwapIOU:

                            printCheck("no intersection")
                            pass
                        else:
                            partialLostTrackers.append(nMouse)
                #If no trackedMice are near and was in entrance area, then do not add to partial lost, they likely left the cage.
                numClose = len(partialLostTrackers) - numClose

                printCheck(numClose, 'nearbyMice')
                if numClose == 0 and tracker.getPosition()[0] > entranceX and tracker.getPosition()[1] > entranceY:
                    event["lost_mouse"].append((frameName, tracker.tag()))
                    printCheck("mouse exited")
                    outRow["Events"] += tracker.tag() + " left the cage"
                    if len(trackedMice) == miceNum:
                        partialLostTrackers.append(tracker)
                    else:
                        if tracker.tag() > 99999:
                            # This is not a dummy tracker, we can find it again
                            lostTrackers.append(tracker)
                        try:
                            trackedMice.remove(tracker)
                        except Exception as e:
                            printCheck(str(e))
                            # for safety idk
                            pass
                elif len(tracker.recordedPositions) <= 5 and tracker.tag() < 99999:
                    printCheck("Mouse is lost")
                    trackedMice.remove(tracker)
                else:
                    partialLostTrackers.append(tracker)

        # ================ Partial Lost Emptying =====================
        if len(partialLostTrackers) > 1:

            # Lost more than one? Then once it reappears we cannot know which it is.
            # We now require the RFID.
            pos = partialLostTrackers[1].getPosition()
            if len(pos) < 6 or len(partialLostTrackers[0].getPosition()) < 6:
                continue
            if partialLostTrackers[0].intersectionOverUnion([pos[0], pos[1], pos[4], pos[5]]) > 0 and len(partialLostTrackers) == 2:
                for tracker in partialLostTrackers:
                    #printCheck(track.tag(), "partial lost")
                    printCheck("Removed ", tracker.tag(), "From partial lost")
                    outRow["Events"] += tracker.tag() + " partial lost expired due to occlusion; "
                    event["lost_mouse"].append((frameName, tracker.tag()))
                    if tracker.tag() > 99999:
                        # This is not a dummy tracker, we can find it again
                        lostTrackers.append(tracker)
                    try:
                        trackedMice.remove(tracker)
                    except ValueError:
                        # for safety idk
                        pass
                partialLostTrackers = []
            elif len(partialLostTrackers) >= 2:
                for i in range( 0, len(partialLostTrackers)):

                    if partialLostTrackers[i].lostCounter == -1:
                        partialLostTrackers[i].lostCounter = 0


                    #if partialLostTrackers[i].lostCounter > 10:
                    #    partialLostTrackers[i].lostCounter = -1;
                    #    if partialLostTrackers[i].tag() > 99999:
                    #        lostTrackers.append(partialLostTrackers[i])






        # ======================================RFID Validation========================================================
            # Check if we can match up a dummy mouse with a tag
        if validationFrame:
            #printCheck(mouse.tag())
            usedIndex = False
            """
            First iterate through all lost trackers. If the pickup is one of
            these, we either have an identification or a dummy swap.
            """
            for tracker in partialLostTrackers:
                if validationTag == tracker.tag():
                    # You have just got a pickup for a mouse that has recently disappeared.
                    # In all likelihood, they are occluded by another, using
                    # this pickup will do nothing but harm,
                    # Ignore this pickup.
                    usedIndex = True
            for tracker in lostTrackers:

                if validationTag == tracker.tag():
                    usedIndex = True
                    # Match!
                    printCheck("validation", validationTag, validationReader)
                    nearestMice = sorted(trackedMice, key= lambda x: x.distanceFromPos(validationReader))
                    if len(nearestMice) < 1 or nearestMice[0].distanceFromPos(validationReader) > 300:
                        # If nearest mouse is not currently detected. do nothing
                        printCheck("no trackedMice near")
                        outRow["Events"] += "Validation not used, no mice near; "
                        break
                    if len(nearestMice) >= 2 and abs(nearestMice[0].distanceFromPos(validationReader) - nearestMice[1].distanceFromPos(validationReader)) < maxSwapDistance:
                        # We cannot be certain which one is over the reader
                        printCheck("Cannot be sure")
                        outRow["Events"] += "Validation not used, ambiguity; "
                        break
                    # Update Dummy Track
                    if nearestMice[0].tag() < 99999:
                        printCheck("found mouse again")
                        outRow["Events"] += "Validated " + validationTag + "in position " + str(nearestAnon.currCoord) + "; "
                        nearestAnon = nearestMice[0]
                        tracker.updatePositions(nearestAnon.recordedPositions)
                        tracker.validate()
                        trackedMice.append(tracker)
                        lostTrackers.remove(tracker)
                        trackedMice.remove(nearestAnon)
                        # DUMMY SWAP
                    else:
                        # There was an identity swap earlier. Correct for it AND increment number of identity swap
                        printCheck("dummy swap")
                        outRow["Events"] += "Validated" + validationTag + "but nearest mouse was not dummy, but " + nearestMice[0].tag() + "- correcting; "
                        event["dummy_swap"].append((frameName, nearestMice[0].tag()))
                        badMouse = nearestMice[0]
                        occlusionEndPoint = badMouse.occlusionPointBefore(list(filter(lambda x: x.tag() != badMouse.tag(), trackedMice)), maxSwapDistance)
                        incorrectPositions = badMouse.trimPositions(occlusionEndPoint)
                        printCheck("occlusion End", occlusionEndPoint)
                        tracker.updatePositions(incorrectPositions)
                        tracker.validate()
                        occlusionStartPoint = badMouse.occlusionPointAfter(list(filter(lambda x: x.tag() != badMouse.tag(), trackedMice)), maxSwapDistance)
                        printCheck("occlusion Start", occlusionStartPoint)
                        badMouse.trimPositions(occlusionStartPoint)
                        trackedMice.append(tracker)
                        lostTrackers.remove(tracker)
                        if badMouse not in lostTrackers:
                            lostTrackers.append(badMouse)
                        trackedMice.remove(badMouse)
            if not usedIndex:
                """
                In this case, the pickup was not one of the lost trackers.
                Therefore, we either have a validation of an existing track
                or an identity swap.
                """
                nearestMice = sorted(trackedMice, key= lambda x: x.distanceFromPos(validationReader))
                actualMouse = list(filter(lambda x: x.tag() == validationTag, trackedMice))[0]
                if len(nearestMice) < 1 or nearestMice[0].distanceFromPos(validationReader) > 300:
                    #If nearest mouse is not currently detected. do nothing
                    outRow["Events"] += "Validation not used, no mice near; "
                    pass
                if len(nearestMice) < 2 or abs(nearestMice[0].distanceFromPos(validationReader) - nearestMice[1].distanceFromPos(validationReader)) > maxSwapDistance:
                    #IDENTITY SWAP
                    if nearestMice[0].tag() != validationTag:
                        #Identity swap, increment number identity swaps
                        # An identity swap has occured. Remove the frames up to the last validation point.
                        #printCheck(list(map(lambda x:x.tag(), trackedMice)))
                        printCheck("identity swap")
                        outRow["Events"] += "Validated" + validationTag + " which is already tracked as " + nearestMice[0].tag() + " - correcting; "
                        event["identity_swap"].append((frameName, [nearestMice[0].tag(), actualMouse.tag()]))
                        #Find first valid point of current track
                        actualOcclusionEndPoint = actualMouse.occlusionPointBefore(list(filter(lambda x: x.tag() != actualMouse.tag(), trackedMice)), maxSwapDistance)
                        nearOcclusionEndPoint = nearestMice[0].occlusionPointBefore(list(filter(lambda x: x.tag() != nearestMice[0].tag(), trackedMice)), maxSwapDistance)
                        nearPositions_new = actualMouse.trimPositions(actualOcclusionEndPoint)
                        actualPositions_new = nearestMice[0].trimPositions(nearOcclusionEndPoint)
                        #Find last valid point of old track
                        nearOcclusionStartPoint = actualMouse.occlusionPointAfter(list(filter(lambda x: x.tag() != actualMouse.tag(), trackedMice)), maxSwapDistance)
                        actualOcclusionStartPoint = nearestMice[0].occlusionPointAfter(list(filter(lambda x: x.tag() != nearestMice[0].tag(), trackedMice)), maxSwapDistance)
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

        # No need to have this as part of validation
        if len(lostTrackers) == 1:
            # Only one lost mouse = only one possibility
            if len(list(filter(lambda x: x.tag() == lostTrackers[0].tag(), trackedMice))) >= 1:
                lostTrackers = []
                error = False
            else:
                printCheck("Only one unknown, assigned")
                outRow["Events"] += "Assigned remaining mouse only remaining tag; "
#                  printCheck(list(map(lambda x: x.tag(), trackedMice)), "trackedMice")
#                 printCheck(list(map(lambda x: x.tag(), lostTrackers)), "lost")
                missingMouse = list(filter(lambda x: x.tag() < 99999, trackedMice))
                if len(missingMouse) > 0:
                    missingMouse = missingMouse[0]
                    lostTrackers[0].updatePositions(missingMouse.recordedPositions)
                    lostTrackers[0].validate()
                    trackedMice.append(lostTrackers[0])
                    lostTrackers = []
                    trackedMice.remove(missingMouse)
                    error = False
        # out, masks = ImageProcessing.cvDrawBoxes(cleanedDetections, image, trackedMice)
        # for tag, mask in masks:
        #     mouse = list(filter(lambda x: x.tag() == tag, trackedMice))
        #     if len(mouse) == 1:
        #         mouse = mouse[0]
        #         mouse.currCoord[2] = {'mask': mask.tolist()}
        outRow["Tracked Mice"] = list(map(lambda x: x.describeSelf(), trackedMice))
        outRow["Lost Tracks"] = list(map(lambda x: x.describeSelf(), lostTrackers))
        outRow["Partially Lost Tracks"] = list(map(lambda x: x.describeSelf(), partialLostTrackers))
        if showVideo:

            cv2.imshow('Demo', out)
            cv2.waitKey(3)
        printCheck(list(map(lambda x: x.tag(), trackedMice)),"trackedMice")
        printCheck(list(map(lambda x: x.tag(), lostTrackers)), "lost")
        outMatrix.append(deepcopy(outRow))
#        time.sleep(0.2)
    mouseDict = {}
    # masksDict = {}
    pbar.close()
    outFrame = pd.DataFrame(outMatrix)
    for mouse in filter(lambda x: x.tag() > 99999, trackedMice + lostTrackers):
        # masksDict.update({mouse.tag(): [x[2] for x in mouse.recordedPositions]})
        for x in mouse.recordedPositions: x[2] = ""
        mouseDict.update({mouse.tag(): mouse.recordedPositions})
        printCheck(mouse.tag(), str(len(mouse.recordedPositions)/frameCount*100) + "% Covered")
    finalOutMatrix = []
    framePointers = {i: 0 for i in mouseDict.keys()}
    for i in range(0, max(map(lambda x: x.recordedPositions[-1][3], mouseDict.values()))):
        finalOutRow = {"Frame": 0, "Mice" = []}
        for tag, m in mouseDict.items():
            if i == m.recordedPositions[framePointers[tag]][3]:
                framePointers[tag] += 1
                finalOutRow["Mice"].append([tag, m.[framePointers[tag]]])
        finalOutMatrix.append(deepcopy(finalOutRow))
    finalOutFrame = pd.DataFrame(finalOutMatrix)

    with pd.ExcelWriter(dataDrive + dataPath + "/live_data.xlsx") as writer:  
        outFrame.to_excel(writer, sheet_name="Live Data")
        finalOutFrame.to_excel(writer, sheet_name="Final Data")
    printCheck(list(map(lambda x: str(x[0]) + ":" + str(len(x[1])), event.items())))
    printCheck("frames without all mice: ", badDetections, badDetections/frameCount*100,  "% error")
    with open(dataDrive + dataPath + "/processed.json", "w") as outfile:
        json.dump(mouseDict, outfile, ensure_ascii=False)
    with open(dataDrive + dataPath + "/masks.json", "w") as outfile:
        json.dump(masksDict, outfile, ensure_ascii=False)


def printCheck(*objects):
    global verbose
    if verbose:
        print(*objects)


def scrapeText(RTS):
    taglist = []
    with open(RTS) as f:
        content = f.readlines()
        for row in content:
            print(row)
            row = row.rstrip("\n")
            taglist.append(int(row))
    return taglist

def run(drive, path, showVid=False, frames=False, verb=True):
    mouseTrackers = []
    global dataPath, dataDrive, useFrames, verbose, tags
    dataDrive = drive
    dataPath = path
    showVideo=showVid
    useFrames=frames
    verbose=verb
    if dataDrive  == "frameData":
        dataFileName = "RTS_test"
        dataFileName += args.get("name", "") + ".txt"
    else:
        dataFileName = dataDrive + dataPath + "/RTS_test.txt"
        print("data file name:", dataFileName)
    tags = scrapeText(dataFileName)
    printCheck(tags)
    for tag in tags:
        mouseTrackers.append(MouseTracker([0, 0], tag))
    file = open(dataFileName, "r")
    RFIDResponses = file.readlines()
    try:
        YOLO("base_tracking", mouseTrackers, [], showVideo)
    except Exception as e:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="1 for showing video, 0 or blank for not")
    ap.add_argument("-f", "--frames", help="Include this argument if you have individual frame files")
    ap.add_argument("-d", "--drive", help="Path to data")
    ap.add_argument("-n", "--name", help="Name of the frame folder/text file")
    ap.add_argument("-l", "--log", help="Include this argument for verbose output")
    args = vars(ap.parse_args())
    showVideo = False
    useFrames = False
    verbose = False
    if args.get("video", None) is not None:
        showVideo = True
    if args.get("frames", None) is not None:
        useFrames = True
    if args.get("log", None) is not None:
        verbose = True
    dataPath = args.get("name", "")
    dataDrive = args.get("drive", "frameData")
    run(dataDrive, dataPath, showVideo, useFrames, verbose)
