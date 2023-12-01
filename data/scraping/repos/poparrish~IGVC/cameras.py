#!/usr/bin/env python
import math
from enum import Enum
from datetime import datetime

import cv2
# temp
import numpy as np
import rospy
from rx import Observable
from std_msgs.msg import String, Int16

import topics
from camera_info import CameraInfo
from camera_msg import CameraMsg
from guidance import contours_to_vectors
from cameraconfig.config import create_persistent_trackbar
from util import rx_subscribe, Vec2d
import os



cam_name = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_2B2150DE-video-index0'
cam_name = '/dev/video1'

# cam_name = '/home/nregner/IGVC/Nav-Guidance/src/navigation_launch/dev/arcs.avi'
# cam_name = '/home/nregner/IGVC/Nav-Guidance/src/navigation_launch/test_videos/overcast_noon_bright128.avi'
#cam_name = '/home/bender/IGVC/Nav-Guidance/src/navigation_launch/speedtest.mp4'

def callback(x):
    #the cv2.createTrackbar() requires callback param
    pass

def process_image(img, camera_info):
    #median blur
    medianRadius = cv2.getTrackbarPos('medianRadius', 'img_medianBlur')
    img_medianBlur = blur(src = img,type = BlurType.Median_Filter, radius = medianRadius)
    cv2.namedWindow('img_medianBlur',0)
    cv2.resizeWindow('img_medianBlur', 640, 480)
    #cv2.imshow('img_medianBlur',img_medianBlur)

    # HSV filter #2 (ramp)
    # RilowH = cv2.getTrackbarPos('RlowH', 'Rimg_HSV')
    # RihighH = cv2.getTrackbarPos('RhighH', 'Rimg_HSV')
    # RilowS = cv2.getTrackbarPos('RlowS', 'Rimg_HSV')
    # RihighS = cv2.getTrackbarPos('RhighS', 'Rimg_HSV')
    # RilowV = cv2.getTrackbarPos('RlowV', 'Rimg_HSV')
    # RihighV = cv2.getTrackbarPos('RhighV', 'Rimg_HSV')
    # hue_threshold = [RilowH, RihighH]
    # sat_threshold = [RilowS, RihighS]
    # val_threshold = [RilowV, RihighV]
    # img_HSV_ramp = hsv_threshold(input=img_medianBlur, hue=hue_threshold, sat=sat_threshold, val=val_threshold)
    # img_HSV = rgb_threshold(img_medianBlur,hue_threshold,sat_threshold,val_threshold)

    #cv2.imshow('Rimg_HSV', img_HSV_ramp)


    #HSV filter #1
    ilowH = cv2.getTrackbarPos('highH','img_HSV')
    ihighH = cv2.getTrackbarPos('lowH','img_HSV')
    ilowS = cv2.getTrackbarPos('lowS','img_HSV')
    ihighS = cv2.getTrackbarPos('highS','img_HSV')
    ilowV = cv2.getTrackbarPos('lowV','img_HSV')
    ihighV = cv2.getTrackbarPos('highV','img_HSV')
    hue_threshold=[ilowH,ihighH]
    sat_threshold=[ilowS,ihighS]
    val_threshold=[ilowV,ihighV]
    img_HSV = hsv_threshold(input=img_medianBlur,hue=hue_threshold,sat=sat_threshold,val=val_threshold)
    # img_HSV = rgb_threshold(img_medianBlur,hue_threshold,sat_threshold,val_threshold)
    img_HSV = hsl_threshold(input=img_medianBlur,h=hue_threshold,l=val_threshold,s=sat_threshold)



    cv2.imshow('img_HSV',img_HSV)

    # apply opening filter to remove ramp lip
    kernel_size = cv2.getTrackbarPos('contoursOpeningKernelSize', 'img_displayFilteredContours')
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_HSV = cv2.morphologyEx(img_HSV, cv2.MORPH_OPEN, kernel)

    #gaussian blur
    gaussianRadius = cv2.getTrackbarPos('gaussianRadius', 'img_gaussianBlur')
    img_gaussianBlur = blur(src=img_HSV,type=BlurType.Gaussian_Blur,radius=gaussianRadius)
    #cv2.imshow('img_gaussianBlur',img_gaussianBlur)

    #birds eye view
    img_displayBirdsEye = camera_info.convertToFlat(img_gaussianBlur)
    #cv2.imshow("birdsEye", img_displayBirdsEye)

    #find contours
    contoursMinArea = cv2.getTrackbarPos('contoursMinArea', 'img_displayFilteredContours')
    contoursMinPerimeter = cv2.getTrackbarPos('contoursMinPerimeter', 'img_displayFilteredContours')
    contoursMinWidth = cv2.getTrackbarPos('contoursMinWidth', 'img_displayFilteredContours')
    contoursMaxWidth = cv2.getTrackbarPos('contoursMaxWidth', 'img_displayFilteredContours')
    contoursMinHeight = cv2.getTrackbarPos('contoursMinHeight', 'img_displayFilteredContours')
    contoursMaxHeight = cv2.getTrackbarPos('contoursMaxHeight', 'img_displayFilteredContours')
    contoursSolidity = [21, 100]
    contoursSolidityMin = cv2.getTrackbarPos('contoursSolidityMin', 'img_displayFilteredContours')
    contoursSolidityMax = cv2.getTrackbarPos('contoursSolidityMax', 'img_displayFilteredContours')
    contoursMaxVertices = cv2.getTrackbarPos('contoursMaxVertices', 'img_displayFilteredContours')
    contoursMinVertices = cv2.getTrackbarPos('contoursMinVertices', 'img_displayFilteredContours')
    contoursMinRatio = cv2.getTrackbarPos('contoursMinRatio', 'img_displayFilteredContours')
    contoursMaxRatio = cv2.getTrackbarPos('contoursMaxRatio', 'img_displayFilteredContours')
    img_rawContours = find_contours(input=img_displayBirdsEye,external_only=False)
    img_displayRawContours=np.ones_like(img_displayBirdsEye)#Return an array of ones with the same shape and type as a given array.
    cv2.drawContours(img_displayRawContours, img_rawContours, -1, (255, 255, 255), thickness=1) #-1 thickness makes them solid
    #cv2.imshow('img_displayRawContours',img_displayRawContours)

    #filter contours
    img_filteredContours = filter_contours(input_contours=img_rawContours,min_area=contoursMinArea,min_perimeter=contoursMinPerimeter,
                                           min_width=contoursMinWidth,max_width=contoursMaxWidth,min_height=contoursMinHeight,
                                           max_height=contoursMaxHeight,solidity=[contoursSolidityMin,contoursSolidityMax],
                                           max_vertex_count=contoursMaxVertices,min_vertex_count=contoursMinVertices,min_ratio=contoursMinRatio,
                                           max_ratio=contoursMaxRatio)
    img_displayFilteredContours = np.ones_like(img)  # Return an array of ones with the same shape and type as a given array.
    cv2.drawContours(img_displayFilteredContours, img_filteredContours, -1, (255, 255, 255), thickness=-1)
    cv2.imshow('img_displayFilteredContours', img_displayFilteredContours)

    return img_displayBirdsEye, img_filteredContours

def filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,min_height, max_height, solidity, max_vertex_count, min_vertex_count,min_ratio, max_ratio):
    output = []
    for contour in input_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w < min_width or w > max_width):
            continue
        if (h < min_height or h > max_height):
            continue
        area = cv2.contourArea(contour)
        if (area < min_area):
            continue
        if (cv2.arcLength(contour, True) < min_perimeter):
            continue
        hull = cv2.convexHull(contour)
        solid = 100 * area / cv2.contourArea(hull)
        if (solid < solidity[0] or solid > solidity[1]):
            continue
        if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
            continue
        ratio = (float)(w) / h
        if (ratio < min_ratio or ratio > max_ratio):
            continue
        output.append(contour)
    return output

def find_contours(input,external_only):
    if (external_only):
        mode = cv2.RETR_EXTERNAL
    else:
        mode = cv2.RETR_LIST
    method = cv2.CHAIN_APPROX_NONE
    im2, contours, hierarchy = cv2.findContours(input, mode=mode, method=method)
    return contours

def hsv_threshold(input, hue, sat, val):
    out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    return cv2.inRange(out, (hue[0], sat[0], val[0]), (hue[1], sat[1], val[1]))

def rgb_threshold(input, r,g,b):
    out = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    return cv2.inRange(out, (r[0], g[0], b[0]), (r[1], g[1], b[1]))

def hsl_threshold(input, h,s,l):
    out = cv2.cvtColor(input, cv2.COLOR_BGR2HLS)
    return cv2.inRange(out, (h[0], s[0], l[0]), (h[1], s[1], l[1]))

def blur(src, type, radius):
    if (type is BlurType.Box_Blur):
        ksize = int(2 * round(radius) + 1)
        return cv2.blur(src, (ksize, ksize))
    elif (type is BlurType.Gaussian_Blur):
        ksize = int(6 * round(radius) + 1)
        return cv2.GaussianBlur(src, (ksize, ksize), round(radius))
    elif (type is BlurType.Median_Filter):
        ksize = int(2 * round(radius) + 1)
        return cv2.medianBlur(src, ksize)
    else:
        return cv2.bilateralFilter(src, -1, round(radius), round(radius))

def convert_to_cartesian(WIDTH,HEIGHT, contours):
    """to merge everything with the lidar we need to set our x origin to the center of the frame and our y origin
    to where the lidar is. this happens to be at the same height in pixels as the blackout box we configure in the
    CameraInfo class.rectangle_height.
    For some fucked up reason the contours that CV outputs has an extra list layer around the individual points...
    the point_ is just going through that layer. so don't let it confuse
    To avoid looping a second time we also have the option to remove 'thin' the contours by removing
    n number of points along the perimiter"""
    for contour in contours:
        for point_ in contour:

            for point in point_:#this is the layer in each contour that has the points
                #shift x to center

                if point[0] > WIDTH/2:
                    point[0] = point[0]-WIDTH/2
                elif point[0] < WIDTH / 2:
                    point[0] = WIDTH / 2 - point[0]
                    point[0] *= -1
                else:
                    point[0] = 0

                #shift y to center
                if point[1] > HEIGHT / 2:
                    point[1] = point[1] - HEIGHT / 2
                    point[1] *= -1
                elif point[1] < HEIGHT / 2:
                    point[1] = HEIGHT / 2 - point[1]
                else:
                    point[1] = 0
        # print contour
    return contours

def contour_slope(contour):
    if contour is None:
        return 0

    x = np.array([v.x for v in contour])
    y = np.array([v.y for v in contour])

    [slope, intercept] = np.polyfit(x, y, 1)
    return math.degrees(math.atan(slope))

def closest_contour_slope(contours):
    i = 0#track which contour
    results = []
    for contour in contours:#find closest point per contour
        closest = 10000000#just a large starting #
        for vec in contour:
            if closest > vec.mag:
                closest = vec.mag
        results.append([i,closest])
        i+=1

    closest_contour = []
    for result in results:
        closest = 10000000
        if closest > result[1]:
            closest = result[1]
            closest_contour = contours[result[0]]

    if len(closest_contour) == 0:
        return 0
    else:
        return contour_slope(closest_contour)

def closest_contour(contours):
    i = 0#track which contour
    results = []
    for contour in contours:#find closest point per contour
        closest = 10000000#just a large starting #
        for vec in contour:
            if closest > vec.mag:
                closest = vec.mag
        results.append([i, closest])
        i+=1

    closest_contour = []
    closest = 10000000
    for result in results:
        if closest > result[1]:
            closest = result[1]
            closest_contour = contours[result[0]]
    if len(closest_contour) == 0:
        return 0
    else:
        return closest_contour

def flatten_contours(pointCloud):
    """merges contours, but preserves grouping. group #'s are arbitrary"""
    cloud = []
    contour_count=0
    for contour in pointCloud:
        contour_count+=1
        for point in contour:
            point.contour_group=contour_count#assign a contour group
            cloud.append(point)
    return cloud,contour_count


def filter_barrel_lines(camera,angle_range,lidar_vecs,mag_cusion,barrel_cusion):
    """
    This removes lines the camera sees that are likely attached to a barrel. It groups all camera data into chunks
    determined by size angle_range. If a laser scan is in the angle_range of the camera chunk and the laser scan is
    in front of the camera chunk it pop() the chunk. mag_cusion allows us some room for lines that may be on top of or
    slightly in front of the laser scan; its intent is to buffer camera vibrations that are independent from the chassis.
    :param camera_vecs: list[] of camera_vec objects
    :param angle_range: int  0-10 range reccomended
    :param lidar_vecs: list[] of lidar_vec objects
    :param mag_cusion: int min 300 reccomended
    :return: camera_vecs input data type with barrel noise removed
    """

    def to360(angle):
        if angle < 0:
            angle = (angle % -360) + 360
        return angle % 360


    check_full = flatten_contours(camera)

    if len(check_full) == 0:
        return check_full
    else:
        return_vecs = []
        # rotate lidar
        for vec in lidar_vecs:
            vec.angle += 90
        # normalize rotation
        for normalized_vec in lidar_vecs:
            new_angle = to360(normalized_vec.angle)
            normalized_vec.angle = new_angle
        # #now 0 degrees is on the right bottom and 180 is on the left bottom.
        for camera_vecs in camera:
            #rotatecamera
            for vec in camera_vecs:
                vec.angle+=90
            #normalize rotation
            for normalized_vec in camera_vecs:
                new_angle=to360(normalized_vec.angle)
                normalized_vec.angle = new_angle


            #now 0 is in front increasing counterclockwise
            camera_vecs.sort(key=lambda x: x.angle)
            start_iter_angle =int(camera_vecs[0].angle)
            end_iter_angle=int(camera_vecs[1].angle)
            try:
                camera_groups = []
                vec_group = []
                camera_vecs_iterator=iter(camera_vecs)
                angle_start = start_iter_angle
                total_dist=0
                while True:
                    next_vec=next(camera_vecs_iterator)
                    if next_vec.angle < angle_start + angle_range:#between i and i+range so add to group
                        vec_group.append(next_vec)
                        total_dist+=next_vec.mag
                    else:
                        avg_dist = total_dist/len(vec_group)
                        camera_groups.append([angle_start,angle_range,avg_dist,vec_group])
                        angle_start+=angle_range
                        vec_group=[]
                        vec_group.append(next_vec)
                        total_dist=next_vec.mag
            except StopIteration:
                pass

            filtered_camera_groups = camera_groups
            for next_lidar_vec in lidar_vecs:
                for camera_group in camera_groups:
                    if int(next_lidar_vec.angle) in range(camera_group[0], camera_group[0] + camera_group[1]):  # check if lidar scan indicates camera data should be thrown
                        #if next_lidar_vec.mag < 2500:#for testing in lab
                        if next_lidar_vec.mag < camera_group[2] + mag_cusion:
                            if camera_group in filtered_camera_groups:
                                filtered_camera_groups.remove(camera_group)
                        break
                    #now check for the pads as determined from the original grouping sort (start_end_iter)
                    if int(next_lidar_vec.angle) in range(int(start_iter_angle), int(start_iter_angle)-barrel_cusion):#right
                        if next_lidar_vec.mag < camera_group[2] + mag_cusion:
                            if camera_group in filtered_camera_groups:
                                filtered_camera_groups.remove(camera_group)
                        break
                    if int(next_lidar_vec.angle) in range(int(end_iter_angle),int(end_iter_angle) + barrel_cusion):#left
                        if next_lidar_vec.mag < camera_group[2] + mag_cusion:
                            if camera_group in filtered_camera_groups:
                                filtered_camera_groups.remove(camera_group)
                        break
            for group in filtered_camera_groups:#remove contour grouping
                for vec in group[3]:
                    return_vecs.append(vec)

        # undo rotation
        for vec in return_vecs:
            vec.angle -= 90
        # normalize rotation
        for normalized_vec in return_vecs:
            new_angle = to360(normalized_vec.angle)
            normalized_vec.angle = new_angle

        for vec in lidar_vecs:
            vec.angle -= 90
        # normalize rotation
        for normalized_vec in lidar_vecs:
            new_angle = to360(normalized_vec.angle)
            normalized_vec.angle = new_angle
        #filtered camera vecs output should just be [vec,vec,vec]


        return return_vecs

def update_lidar(laser_pickle):
    global lidar
    lidar=laser_pickle

def vector_to_point(vector):
    return vector.x, vector.y

def vectors_to_points(vector_contours):
    return [[vector_to_point(v) for v in c]for c in vector_contours]

def vectors_to_contours(vectors):#same as points just restores contour grouping
    if len(vectors) ==0:
        return []
    i =0
    #sort contours by contour group
    vectors_sorted=sorted(vectors,key=lambda x: x.contour_group)
    num_vectors = len(vectors_sorted)-1
    num_contours=vectors_sorted[num_vectors].contour_group

    contours = [[]for j in range(num_contours)]#init a 2d list for contours
    for v in vectors_sorted:
        if v.contour_group == i+1:
            contours[i].append(v.with_angle(v.angle))
        else:
            i+=1

    return contours

def calculate_line_angle(contour):
    if contour is None:
        return 0

    x = np.array([v.x for v in contour])
    y = np.array([v.y for v in contour])

    [slope, intercept] = np.polyfit(x, y, 1)
    return math.degrees(math.atan(slope)),slope,intercept


def update_exposure(value):
    res = os.system('v4l2-ctl --device=' + cam_name + ' --set-ctrl=exposure_auto=1 && ' +
                    'v4l2-ctl --device=' + cam_name + ' --set-ctrl=exposure_absolute=' + str(value))
    rospy.loginfo('Updated exposure ' + str(value) + ' ' + str(res))


def update_auto_white(white):
    res=os.system('v4l2-ctl --device=' + cam_name + ' --set-ctrl=white_balance_temperature_auto=0 && ' +
                  'v4l2-ctl --device=' + cam_name + ' --set-ctrl=white_balance_temperature=' + str(white))
    rospy.loginfo('Updated auto_white'+str(white)+' '+str(res))

def camera_processor():


    # open a video capture feed
    cam = cv2.VideoCapture(cam_name)

    #init ros & camera stuff
    # pub = rospy.Publisher(topics.CAMERA, String, queue_size=10)
    no_barrel_pub=rospy.Publisher(topics.CAMERA,String, queue_size=10)
    line_angle_pub=rospy.Publisher(topics.LINE_ANGLE, Int16, queue_size=0)
    global lidar
    lidar_obs = rx_subscribe(topics.LIDAR)

    Observable.combine_latest(lidar_obs, lambda n: (n)) \
        .subscribe(update_lidar)

    rospy.init_node('camera')
    rate = rospy.Rate(10)

    exposure_init = False

    rawWidth = 640
    rawHeight = 480
    #camera_info = CameraInfo(53,38,76,91,134)#ground level#half (134 inches out)
    camera_info = CameraInfo(53,40,76,180,217,croppedWidth,croppedHeight)#ground level# 3/4 out
    while not rospy.is_shutdown():


        #grab a frame
        ret_val, img = cam.read()

        # camera will set its own exposure after the first frame, regardless of mode
        if not exposure_init:
            update_exposure(cv2.getTrackbarPos('exposure', 'img_HSV'))
            update_auto_white(cv2.getTrackbarPos('auto_white','img_HSV'))
            exposure_init = True

        #record a video simultaneously while processing
        if ret_val==True:
            out.write(img)
	
        #for debugging
        # cv2.line(img,(640/2,0),(640/2,480),color=(255,0,0),thickness=2)
        # cv2.line(img,(0,int(480*.25)),(640,int(480*.25)),color=(255,0,0),thickness=2)

        #crop down to speed processing time
        #img = cv2.imread('test_im2.jpg')
        dim = (rawWidth,rawHeight)
        img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
        cropRatio = float(croppedHeight)/float(rawHeight)
        crop_img = img[int(rawHeight * float(1-cropRatio)):rawHeight, 0:rawWidth]  # crops off the top 25% of the image
        cv2.imshow("cropped", crop_img)

        #process the cropped image. returns a "birds eye" of the contours & binary image
        img_displayBirdsEye, contours = process_image(crop_img, camera_info)

        #raw
        contours = convert_to_cartesian(camera_info.map_width, camera_info.map_height, contours)
        #for filtered barrels
        vec2d_contour = contours_to_vectors(contours)#replaces NAV
        filtered_contours = filter_barrel_lines(camera=vec2d_contour, angle_range=8,lidar_vecs=lidar,mag_cusion=300,barrel_cusion=5)

        #EXTEND THE LINES
        filtered_cartesian_contours = vectors_to_contours(filtered_contours)

        try:

            closest_filtered_contour = closest_contour(filtered_cartesian_contours)

            # print "CLOSESTCONTOUR: ",closest_filtered_contour

            x_range = 5000
            contour_lines=[]
            interval = 40

            #just one
            line_angle, slope, intercept = calculate_line_angle(closest_filtered_contour)
            for x in range(x_range * -1, x_range):
                if x % interval == 0:
                    y = slope * x + intercept
                    v = Vec2d.from_point(x, y)
                    contour_lines.append(v)

        except TypeError:#no camera data
            contour_lines=[]
            line_angle=0


        #build the camera message with the contours and binary image
        # local_map_msg = CameraMsg(contours=contours, camera_info=camera_info)
        # filtered_map_msg=CameraMsg(contours=contour_lines,camera_info=camera_info)#1 polyfit contour
        c = []
        for cs in filtered_cartesian_contours:
            for v in cs:
                c.append(v)
        filtered_map_msg=CameraMsg(contours=c,camera_info=camera_info)#all raw contours


        #make bytestream and pass if off to ros
        # local_map_msg_string = local_map_msg.pickleMe()
        filtered_map_msg_string=filtered_map_msg.pickleMe()

        #rospy.loginfo(local_map_msg_string)
        # pub.publish(local_map_msg_string)
        no_barrel_pub.publish(filtered_map_msg_string)
        line_angle_pub.publish(line_angle)


        if cv2.waitKey(1) == 27:
            break
        rate.sleep()
    cv2.destroyAllWindows()





if __name__ == '__main__':

    # plt.ion()
    # plt.axis([0, 640, 360, 0])
    # animated_plot=plt.plot([],[],'ro')[0]


    # initial values for filters
    BlurType = Enum('BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')

    # Median blur
    cv2.namedWindow('img_medianBlur')
    medianRadius = 7
    cv2.createTrackbar('medianRadius', 'img_medianBlur', medianRadius, 20, callback)

    # HSV
    cv2.namedWindow('img_HSV')
    cv2.namedWindow('Rimg_HSV')

    ilowH = 0
    ihighH = 60
    ilowS = 41
    ihighS = 64
    ilowV = 0
    ihighV = 210

    ilowH = 39
    ihighH = 86
    ilowS = 27
    ihighS = 156
    ilowV = 93
    ihighV = 255

    #poolnoodle
    # ilowH = 45
    # ihighH = 76
    # ilowS = 28
    # ihighS = 156
    # ilowV = 160
    # ihighV = 252

    # bright 128 sunny (best, sorts into blue hue, exposure at shiniest thing it will see)
    ilowH = 135
    ihighH = 255
    ilowS = 186
    ihighS = 255
    ilowV = 172
    ihighV = 255

    RilowH = 114
    RihighH = 149
    RilowS = 85
    RihighS = 255
    RilowV = 28
    RihighV = 189

    create_persistent_trackbar('lowH', 'img_HSV', ilowH, 255)
    create_persistent_trackbar('highH', 'img_HSV', ihighH, 255)
    create_persistent_trackbar('lowS', 'img_HSV', ilowS, 255)
    create_persistent_trackbar('highS', 'img_HSV', ihighS, 255)
    create_persistent_trackbar('lowV', 'img_HSV', ilowV, 255)
    create_persistent_trackbar('highV', 'img_HSV', ihighV, 255)
    create_persistent_trackbar('exposure', 'img_HSV', 3, 255, update_exposure)
    create_persistent_trackbar('auto_white','img_HSV',0,6000, update_auto_white)

    cv2.createTrackbar('RlowH', 'Rimg_HSV', RilowH, 255, callback)
    cv2.createTrackbar('RhighH', 'Rimg_HSV', RihighH, 255, callback)
    cv2.createTrackbar('RlowS', 'Rimg_HSV', RilowS, 255, callback)
    cv2.createTrackbar('RhighS', 'Rimg_HSV', RihighS, 255, callback)
    cv2.createTrackbar('RlowV', 'Rimg_HSV', RilowV, 255, callback)
    cv2.createTrackbar('RhighV', 'Rimg_HSV', RihighV, 255, callback)

    # Gaussian Blur
    cv2.namedWindow('img_gaussianBlur')
    gaussianRadius = 2
    cv2.createTrackbar('gaussianRadius','img_gaussianBlur',gaussianRadius, 20,callback)

    # filter contours
    cv2.namedWindow('img_displayFilteredContours')
    contoursMinArea = 1000
    contoursMinPerimeter = 1
    contoursMinWidth = 0
    contoursMaxWidth = 1000000
    contoursMinHeight = 0
    contoursMaxHeight = 1000000
    contoursSolidity = [21, 100]
    contoursSolidityMin = 21
    contoursSolidityMax = 100
    contoursMaxVertices = 1000000
    contoursMinVertices = 0
    contoursMinRatio = 0
    contoursMaxRatio = 10000

    cv2.createTrackbar('contoursMinArea','img_displayFilteredContours',contoursMinArea,50000, callback)
    cv2.createTrackbar('contoursMinPerimeter','img_displayFilteredContours',contoursMinPerimeter,2000, callback)
    cv2.createTrackbar('contoursMinWidth','img_displayFilteredContours',contoursMinWidth,1000, callback)
    cv2.createTrackbar('contoursMaxWidth','img_displayFilteredContours',contoursMaxWidth,1000, callback)
    cv2.createTrackbar('contoursMinHeight','img_displayFilteredContours',contoursMinHeight,1000, callback)
    cv2.createTrackbar('contoursMaxHeight','img_displayFilteredContours',contoursMaxHeight,1000, callback)
    cv2.createTrackbar('contoursSolidityMin','img_displayFilteredContours',contoursSolidityMin,100, callback)
    cv2.createTrackbar('contoursSolidityMax','img_displayFilteredContours',contoursSolidityMax,100, callback)
    cv2.createTrackbar('contoursMaxVertices','img_displayFilteredContours',contoursMaxVertices,2000, callback)
    cv2.createTrackbar('contoursMinVertices','img_displayFilteredContours',contoursMinVertices,2000, callback)
    cv2.createTrackbar('contoursMinRatio','img_displayFilteredContours',contoursMinRatio,100, callback)
    cv2.createTrackbar('contoursMaxRatio','img_displayFilteredContours',contoursMaxRatio,100, callback)
    create_persistent_trackbar('contoursOpeningKernelSize', 'img_displayFilteredContours', 0)

    # create trackBars for on real time tuning

    croppedWidth = 640
    croppedHeight = 360

    

    lidar = []

    #initialize the camera video recorder
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    extension = '.avi'
    name=str(datetime.now())
    file_name=name+extension
    out = cv2.VideoWriter(file_name,fourcc,10.0,(640,480))

    try:
        camera_processor()
    except rospy.ROSInterruptException:
        pass
