# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import os
from multiprocessing import Pipe, Process

RESOLUTION = (320, 240)
CENTER = (RESOLUTION[0] / 2, RESOLUTION[1] / 2)  # 160, 120
CHANNEL_MAP = {11: 17}


class PWD:
    __slots__ = ('pin', 'pos')

    def __init__(self, pin):
        self.pin = pin
        self.set(50)

    def set(self, pos):
        self.pos = pos
        cmd = 'echo "%d=%.2f" > /dev/pi-blaster' % (CHANNEL_MAP[self.pin], pos/100.)
        os.system(cmd)

    def move(self, value):
        position = self.pos + value

        if position > 100:
            position = 100
        if position <= 10:
            position = 10
        self.set(position)



def locate_ball(pipe):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    ap.add_argument("-p", "--picamera", type=int, default=-1,
                    help="whether or not the Raspberry Pi camera should be used")
    args = vars(ap.parse_args())

    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space, then initialize the
    # list of tracked points
    greenLower = (37, 71, 13)
    greenUpper = (79, 255, 255)
    pts = deque(maxlen=args["buffer"])

    # if a video path was not supplied, grab the reference
    # to the webcam
    if not args.get("video", False):
        vs = VideoStream(usePiCamera=args["picamera"] > 0, resolution=RESOLUTION).start()

    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])


    # allow the camera or video file to warm up
    time.sleep(2.0)

    # keep looping
    while True:
        # grab the current frame
        frame = vs.read()

        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if args.get("video", False) else frame

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None
        radius = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # update the points queue
        pts.appendleft(center)

        pipe.send((center, radius))

        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    # if we are not using a video file, stop the camera video stream
    if not args.get("video", False):
        vs.stop()

    # otherwise, release the camera
    else:
        vs.release()

    # close all windows
    cv2.destroyAllWindows()
    pipe.close()


def print_coords(pipe, y_servo):
    x_center, y_center = CENTER
    scr_width, scr_height = RESOLUTION
    while True:
        coords, radius = pipe.recv()
        if coords and radius:
            x, y = coords
            delta_x = x_center - x
            delta_y = y_center - y

            rel_x = delta_x / scr_width * 100
            rel_y = delta_y / scr_height * 100

            print(delta_x, delta_y)

            # x_servo.move(rel_x)
            y_servo.move(rel_y)







if __name__ == '__main__':
    y = PWD(11)
    receiver, sender = Pipe(duplex=False)

    ball_catcher = Process(target=locate_ball, args=(sender,))
    coord_writer = Process(target=print_coords, args=(receiver, y))

    ball_catcher.start()
    coord_writer.start()

    ball_catcher.join()
    coord_writer.join()
