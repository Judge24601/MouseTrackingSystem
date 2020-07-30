"""
Tracking file. Copy into the darknet folder for processing.
"""

import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter
import cv2

class MouseTracker:
    totalAllowedVis = 8
    kVelocityDiff = 0.01

    def __init__(self, startCoord, id, mask = [], frameCount = 0, width = 0, height = 0):
        self.currCoord = startCoord
        self.positionQueue = deque(maxlen=2)
        if startCoord != [0,0]:
            startCoord.append(mask)
            startCoord.append(frameCount)
            startCoord.append(width)
            startCoord.append(height)
            self.lastPos = None
            self.recordedPositions = [startCoord]
        else:
            self.recordedPositions = []
            self.lastPos = None
        self.visualTracker = None
        self.visualCount = 0
        self.canDoVisual = False
        self.validatedIndex = 0
        self.id = id
        self.lastFrameCount = 0
        self.velocity = (0,0)
        self.lostCounter= -1

    def updatePosition(self, coordinate, mask=[], frameCount = 0, width = 0, height = 0):
        self.currCoord = coordinate
        coordinate.append(mask)
        coordinate.append(frameCount)
        coordinate.append(width)
        coordinate.append(height)
        if (frameCount - self.lastFrameCount) > 5:
            self.lastPos = None
        self.lastFrameCount = frameCount
        self.recordedPositions.append(coordinate)
        if len(self.recordedPositions) > 10:
            self.canDoVisual = True
        if self.lastPos and len(coordinate) > 1:
            self.velocity = ((coordinate[0] - self.lastPos[0]), (coordinate[1] - self.lastPos[1]))
        else:
            self.velocity = (0,0)
        self.lastPos = coordinate
        if self.visualTracker is not None:
            self.visualCount += 1
            if self.visualCount > MouseTracker.totalAllowedVis:
                self.stopVisualTracking()

    def startVisualTracking(self, frame):
        self.visualTracker = cv2.TrackerCSRT_create()
        if len(self.currCoord) <= 5:
            return False
        bbox = (self.currCoord[0] - self.currCoord[4]/2,
            self.currCoord[1] - self.currCoord[5]/2,
            self.currCoord[4], self.currCoord[5])
        if not self.visualTracker.init(frame, bbox):
            self.visualTracker = None
            return False
        else:
            self.visualStartPoint = self.currCoord
            self.canDoVisual = False
            return True

    def stopVisualTracking(self, delete=True):
        self.visualTracker = None
        self.visualCount = 0
        if delete:
            self.trimPositions(self.visualStartPoint[3])
            self.canDoVisual = False
        else:
            self.canDoVisual = True

    def validate(self):
        self.validatedIndex = len(self.recordedPositions) - 1

    def lastValidatedPosition(self):
        if self.validatedIndex > 0 and len(self.recordedPositions) > self.validatedIndex:
            return self.recordedPositions[self.validatedIndex]
        if len(self.recordedPositions) > 0:
            return self.recordedPositions[0]
        return [0, 0, -1, 0, 0, 0]

    def updatePositions(self, newPositions):
        for position in newPositions:
            for rec in self.recordedPositions:
                if rec[3] == position[3]:
                    self.recordedPositions.remove(rec)
        self.recordedPositions.extend(newPositions)
        self.recordedPositions = sorted(self.recordedPositions, key = lambda x: x[3])
        self.validatedIndex = len(self.recordedPositions) - 1
        if self.validatedIndex >= 0:
           self.currCoord = self.recordedPositions[self.validatedIndex]

    def trimPositions(self, frameCount = 0):
        index = None
        for rec in self.recordedPositions:
            if rec[3] >= frameCount:
                index = self.recordedPositions.index(rec)
                break
        if index and index > self.validatedIndex:
            self.validatedIndex = index
        tempPositions = self.recordedPositions[self.validatedIndex + 1:]
        self.recordedPositions = self.recordedPositions[:self.validatedIndex+1]
        if self.validatedIndex >= 0 and len(self.recordedPositions) > self.validatedIndex:
           self.currCoord = self.recordedPositions[self.validatedIndex]
       # print(self.id, "Trimmed to", self.validatedIndex, self.lastValidatedPosition())
        return tempPositions

    def getPosition(self):
        return self.currCoord

    def describeSelf(self):
        try:
            return [self.id, self.currCoord[0], self.currCoord[1], self.currCoord[4], self.currCoord[5]]
        except IndexError:
            try:
                return [self.id, self.currCoord[0], self.currCoord[1]]
            except IndexError:
                return [self.id]

    def distanceFromPos(self, pos):
        x1 = self.currCoord[0]
        y1 = self.currCoord[1]
        x2 = pos[0]
        y2 = pos[1]
        return np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

    def intersectionOverUnion (self, pos):
            if len(self.currCoord) < 5:
                return 0
            selfArea = self.currCoord[4]*self.currCoord[5]
            if selfArea == 0:
                return 0
            posArea = pos[2]*pos[3]
            selfMinX = self.currCoord[0] - self.currCoord[4]/2
            selfMaxX = self.currCoord[0] + self.currCoord[4]/2
            selfMinY = self.currCoord[1] - self.currCoord[5]/2
            selfMaxY = self.currCoord[1] + self.currCoord[5]/2

            otherMinX = pos[0] - pos[2]/2
            otherMaxX = pos[0] + pos[2]/2
            otherMinY = pos[1] - pos[3]/2
            otherMaxY = pos[1] + pos[3]/2

            # Compute the intersection boundaries
            interLeft   = max(selfMinX, otherMinX);
            interTop    = max(selfMinY, otherMinY);
            interRight  = min(selfMaxX, otherMaxX);
            interBottom = min(selfMaxY, otherMaxY);
            # If the intersection is valid (positive non zero area), then there is an intersection
            if ((interLeft < interRight) and (interTop < interBottom)):
                intersection = (interRight - interLeft)*(interBottom - interTop)
            else:
                intersection = 0
            union = posArea + selfArea - intersection
            return intersection/union

    def trackLikelihood (self, pos, image):
        IOU = self.intersectionOverUnion(pos)
        new_vel = (pos[0] - self.currCoord[0], pos[1] - self.currCoord[1])
        del_vel = np.sqrt((new_vel[0] - self.velocity[0])**2 +
                    (new_vel[1] - self.velocity[1])**2)
        return IOU - MouseTracker.kVelocityDiff*del_vel

    def tag(self):
        return self.id

    def occlusionPointBefore(self, others, distance):
        """
        Examines the recorded positions of the mouse and compares them
        to all other mice locations. The last known point where this mouse
        was near to another will be defined as the end of its last occlusion.
        """
        lastCheckedFrameDict = {}
        for mouse in others:
            lastCheckedFrameDict.update({mouse.tag(): len(mouse.recordedPositions) -1})
        occlusionPoint = len(self.recordedPositions) -1
        if occlusionPoint < 0:
            return 0
        endLoop = False
        for i in range(len(self.recordedPositions) -1, -1, -1):
            for mouse in others:
                if lastCheckedFrameDict[mouse.tag()] < 0:
                    continue
                while self.recordedPositions[i][3] < mouse.recordedPositions[lastCheckedFrameDict[mouse.tag()]][3]:
                    if lastCheckedFrameDict[mouse.tag()] > 0:
                        lastCheckedFrameDict[mouse.tag()] -= 1
                    else:
                        break
                if self.recordedPositions[i][3] == mouse.recordedPositions[lastCheckedFrameDict[mouse.tag()]][3]:
                    print(self.recordedPositions[i][3])
                    x1, y1 = self.recordedPositions[i][0], self.recordedPositions[i][1]
                    x2, y2 = mouse.recordedPositions[lastCheckedFrameDict[mouse.tag()]][0], mouse.recordedPositions[lastCheckedFrameDict[mouse.tag()]][1]
                    if (np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))) <= distance:
                        print("near to other")
                        occlusionPoint = self.recordedPositions[i][3]
                        endLoop = True
                        break
            if endLoop:
                break
        return occlusionPoint

    def occlusionPointAfter(self, others, distance):
        """
        Examines the recorded positions of the mouse and compares them
        to all other mice locations. The first point after the last validation
        of this mouse where it is near to another will be defined
        as the beginning of its first occlusion.
        """
        lastCheckedFrameDict = {}
        if len(self.recordedPositions) == 0:
            return 0
        for mouse in others:
            lastCheckedFrameDict.update({mouse.tag(): 0})
        occlusionPoint = self.validatedIndex
        if occlusionPoint >= len(self.recordedPositions):
            return len(self.recordedPositions) - 1
        endLoop = False
        for i in range(self.validatedIndex, len(self.recordedPositions)):
            for mouse in others:
                if len(mouse.recordedPositions) <= lastCheckedFrameDict[mouse.tag()]:
                    continue
                while self.recordedPositions[i][3] > mouse.recordedPositions[lastCheckedFrameDict[mouse.tag()]][3]:
                    if lastCheckedFrameDict[mouse.tag()] < len(mouse.recordedPositions) -1:
                        lastCheckedFrameDict[mouse.tag()] += 1
                    else:
                        break
                if self.recordedPositions[i][3] == mouse.recordedPositions[lastCheckedFrameDict[mouse.tag()]][3]:
                    x1, y1 = self.recordedPositions[i][0], self.recordedPositions[i][1]
                    x2, y2 = mouse.recordedPositions[lastCheckedFrameDict[mouse.tag()]][0], mouse.recordedPositions[lastCheckedFrameDict[mouse.tag()]][1]
                    if (np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))) <= distance:
                        occlusionPoint = self.recordedPositions[i][3]
                        endLoop = True
                        break
            if endLoop:
                break
        return occlusionPoint
        pass
