#!usr/bin/python
#-- coding: utf8 -- 
import cv2 as cv
import numpy as np
import math
#import torch
#import torch.nn as nn


'''
Helper functions begin
'''


def void_callback(x):
    pass


def DrawRotatedRect(img, rect, color=(0, 255, 0), thickness=2):
    center = (int(rect[0][0]),int(rect[0][1]))
    angle = rect[2]
    font = cv.FONT_HERSHEY_COMPLEX
    #cv.putText(img, str(angle), center, font, 0.5, color, thickness, 8, 0)
    vertices = cv.boxPoints(rect)
    for i in range(4):
        cv.line(img, tuple(vertices[i]), tuple(vertices[(i + 1) % 4]), color, thickness)
    return img


def formatPrint(title, items, filename):
    try:
        file = open(filename, 'w')
    except:
        print('cannot Open the file')
        return
    file.writelines(title + ' {' + '\n')
    for i in items:
        file.writelines('  ' + str(i) + '\n')
    file.writelines('}' + '\n')
    file.close()


def pointCmp(p1, p2):
    if p1[0] > p2[0]:
        return 1
    elif p1[0] == p2[0]:
        return 0
    else:
        return -1


def armorCmp(a1, a2):
    if a1.area > a2.area:
        return 1
    elif a1.area == a2.area:
        return 0
    else:
        return -1


def solveArmorCoordinate(width, height):
    return [(-width / 2, height / 2, 0.0), (width / 2, height / 2, 0.0), (width/2,  -height/2,  0.0), (-width/2, -height/2, 0.0)]


'''
Helper functions end
'''

'''
Intermediate Classes Begin
'''


class LightBar:
    def __init__(self, vertices):
        # The length of edges
        edge1 = np.linalg.norm(vertices[0] - vertices[1])
        edge2 = np.linalg.norm(vertices[1] - vertices[2])
        if edge1 > edge2:
            self._width = edge1
            self._height = edge2
            if vertices[0][1] < vertices[1][1]:
                self._angle = math.atan2(vertices[1][1] - vertices[0][1], vertices[1][0] - vertices[0][0])
            else:
                self._angle = math.atan2(vertices[0][1] - vertices[1][1], vertices[0][0] - vertices[1][0])
        else:
            self._width = edge2
            self._height = edge1
            if vertices[2][1] < vertices[1][1]:
                self._angle = math.atan2(vertices[1][1] - vertices[2][1], vertices[1][0] - vertices[2][0])
            else:
                self._angle = math.atan2(vertices[2][1] - vertices[1][1], vertices[2][0] - vertices[1][0])
        # Convert to degree
        self.angle = (self._angle * 180) / math.pi
        self.area = self._width * self._height
        self.aspect_ratio = self._width / self._height
        self.center = (vertices[1] - vertices[3]) / 2
        self.vertices = vertices[:]  # Create a copy instead of a reference


class Armor:
    def __init__(self, armor_rect, armor_vertex, armor_stddev=0.0):
        self.rect = armor_rect
        self.vertex = armor_vertex
        self.stddev = armor_stddev
        self.area = armor_rect[1][0] * armor_rect[1][1]

'''
Intermediate Classes End
'''

'''
Process Classes Begin Template Provided
'''
class GrayImageProc:
    def __call__(self, image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

class HSVImageProc:
    def __init__(self, enable_debug=True, color='blue', ranges=None):
        self.enable_debug = enable_debug
        self._color = color
        if ranges is None:
            if self._color == 'blue':
                self._ranges = [90, 150, 46, 240, 255, 255]
            else:
                self._ranges = [170, 43, 46, 3, 255, 255]
        else:
            self._ranges = ranges
        if enable_debug:
            cv.namedWindow('image_proc')
            self.bars_name = ['h_low', 's_low', 'v_low', 'h_high', 's_high', 'v_high']
            self._bars = [
                cv.createTrackbar(self.bars_name[i], 'image_proc', 0, 255 if i % 3 != 0 else 180, void_callback) for i
                in range(6)]

    def Update(self):
        if self.enable_debug:
            for i in range(6):
                self._ranges[i] = cv.getTrackbarPos(self.bars_name[i], 'image_proc')
        else:
            print("Not On debug Mode!")

    def __call__(self, img):
        self.Update()
        element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img = cv.dilate(img, element, anchor=(-1, -1), iterations=1)
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower = self._ranges[:3]
        upper = self._ranges[3:]
        if lower[0] > upper[0]:
            thresh1_img = cv.inRange(hsv_img, [0] + lower[1:], upper)
            thresh2_img = cv.inRange(hsv_img, lower, [180] + upper[1:])
            thresh_img = thresh1_img | thresh2_img
        else:
            thresh_img = cv.inRange(hsv_img, lower, upper)
        if self.enable_debug:
            cv.imshow('thresholded', thresh_img)
        return thresh_img

class BGRImageProc:
    '''
    color: B: Blue; R: Red.
    threshs: [b-r, b-g]; [r-b, r-g]
    '''
    def __init__(self, color='B', threshs=None, enable_debug=True):
        if threshs is None:
            self._threshs = [10, 10]
        else:
            self._threshs = threshs
        self._color = color
        self.enable_debug = enable_debug
        if enable_debug:
            cv.createTrackbar('Thresh1', 'image_proc', 0, 255, void_callback)
            cv.createTrackbar('Thresh2', 'image_proc', 0, 255, void_callback)

    def Update(self):
        '''self._threshs[0] = cv.getTrackbarPos('Thresh1', 'image_proc')
        self._threshs[1] = cv.getTrackbarPos('Thresh2', 'image_proc')'''

    def __str__(self):
        return "rgb_threshold1: " + str(self._threshs[0]) + '\n' + "rgb_threshold2: " + str(self._threshs[1])

    def __call__(self, img):
        # Feature enhance
        self.Update()
        element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img = cv.dilate(img, element, anchor=(-1, -1), iterations=1)
        if self._color == 'B':
            b_r = cv.subtract(img[:, :, 0].img[:, :, 2])
            _, b_r = cv.threshold(img, self._threshs[0], 255, cv.THRESH_BINARY)
            b_g = cv.subtract(img[:, :, 0].img[:, :, 1])
            _, b_g = cv.threshold(img, self._threshs[1], 255, cv.THRESH_BINARY)
            thresh_img = b_g & b_r
        else:
            r_b = cv.subtract(img[:, :, 2],img[:, :, 0])
            #cv.imshow("rb_thresh", r_b)
            _, r_b = cv.threshold(r_b, self._threshs[0], 255, cv.THRESH_BINARY)
            r_g = cv.subtract(img[:, :, 2],img[:, :, 1])
            #cv.imshow("rg_thresh",r_g)
            _, r_g = cv.threshold(r_g, self._threshs[1], 255, cv.THRESH_BINARY)
            thresh_img = r_b & r_g
        if self.enable_debug:
            cv.imshow("Threshed Image", thresh_img)
        return thresh_img

class ScreenLightBars:
    def __init__(self,thresh=20 ,enable_debug=False):
        # Create Rectangular 
        self._element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        #self._mode = mode
        #cv.createTrackbar("Color", "image_proc", 0, 255, void_callback)
        # Need to define file read action
        self._threshold = thresh
        self._enable_debug = enable_debug

    def Update(self):
        pass
        #self._threshold = cv.getTrackbarPos('Color', 'image_proc')

    def __str__(self):
        return "color_thread: " + str(self._threshold)

    def __call__(self, thresh_img, gray_img, src):
        self.Update()
        cv.imshow("thresh",thresh_img)
        src = src[:]
        light_bars = []
        brightness = cv.threshold(gray_img, self._threshold, 255, cv.THRESH_BINARY)
        
        cv.imshow("brightness", brightness[1])
        
        light_cnts, _ = cv.findContours(brightness[1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        color_cnts, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) ###
        
        for i in light_cnts:
            for j in color_cnts:
                if cv.pointPolygonTest(j, (i[0][0][0], i[0][0][1]), False) >= 0.0: ###
                    single_light = cv.minAreaRect(i)
                    vertices = cv.boxPoints(single_light)  # corner points
                    new_lb = LightBar(vertices)
                    single_light = list(single_light)
                    single_light[2] = new_lb.angle  # Modify the angle
                    single_light = tuple(single_light) ###
                    light_bars.append(single_light)
                    if self._enable_debug:
                        src = DrawRotatedRect(src, single_light)
        if self._enable_debug:
            cv.imshow('light_bars', src)
        return light_bars, src

class FilterLightBars:
    def __init__(self, light_max_aspect_ratio, light_min_area, enable_debug=False):
        self._light_max_aspect_ratio = light_max_aspect_ratio
        self._light_min_area = light_min_area
        self._enable_debug = enable_debug

    def __call__(self, light_bars, src):
        rects = []
        for light_bar in light_bars:
            if ((light_bar[2] != 0) and (light_bar[1][1] != 0)): ###
                vertices = cv.boxPoints(light_bar)
                new_lb = LightBar(vertices)
                area = new_lb.area
                width = light_bar[1][0]
                height = light_bar[1][1]
                light_aspect_ratio = max(width, height) / min(width, height)
                if light_aspect_ratio < self._light_max_aspect_ratio and area >= self._light_min_area:
                    rects.append(light_bar)
                    if self._enable_debug:
                        pass #src = DrawRotatedRect(src, light_bar)
        if self._enable_debug:
            cv.imshow('light_bars_filtered', src)
        return rects

class PossibleArmors:
    def __init__(self, light_max_angle_diff, armor_max_aspect_ratio, armor_min_area, armor_max_pixel_val,
                 enable_debug=False):
        self._light_max_angle_diff = light_max_angle_diff
        self._armor_max_aspect_ratio = armor_max_aspect_ratio
        self._armor_min_area = armor_min_area
        self._armor_max_pixel_val = armor_max_pixel_val
        self._enable_debug = enable_debug

    def calcArmorInfo(self, left_light, right_light):
        armor_points = []
        left_points = cv.boxPoints(left_light)
        right_points = cv.boxPoints(right_light)
        sorted(left_points, key=cmp_to_key(pointCmp))
        sorted(right_points, key=cmp_to_key(pointCmp))
        if right_points[0][1] < right_points[1][1]:
            right_lu = right_points[0]
            right_ld = right_points[1]
        else:
            right_lu = right_points[1]
            right_ld = right_points[0]

        if left_points[2][1] < left_points[3][1]:
            lift_ru = left_points[2]
            lift_rd = left_points[3]
        else:
            lift_ru = left_points[3]
            lift_rd = left_points[2]
        armor_points.append(lift_ru)
        armor_points.append(right_lu)
        armor_points.append(right_ld)
        armor_points.append(lift_rd)
        return armor_points

    def __call__(self, rects, src):
        armors = []
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                rect1 = rects[i]
                rect2 = rects[j]
                edge1min = min(rect1[1][0], rect1[1][1])
                edge1max = max(rect1[1][0], rect1[1][1])
                edge2min = min(rect2[1][0], rect2[1][1])
                edge2max = max(rect2[1][0], rect2[1][1])
                lights_dis = math.sqrt(math.pow(rect1[0][0] - rect2[0][0], 2) + math.pow(rect1[0][1] - rect2[0][1], 2))
                center_angle = math.atan2(abs(rect1[0][1] - rect2[0][1]), abs(rect1[0][0] - rect2[0][0])) * 180 / np.pi
                if center_angle > 90:
                    center_angle = 180 - center_angle

                x = (rect1[0][0] + rect2[0][0]) / 2
                y = (rect1[0][1] + rect2[0][1]) / 2
                width = abs(lights_dis - max(edge1min, edge2min))
                height = max(edge1max, edge2max)
                rect_width = max(width, height)
                rect_height = min(width, height)
                rect = ((x, y), (rect_width, rect_height+30), center_angle)

            
                rect1_angle = rect1[2]
                rect2_angle = rect2[2]

                radio = max(edge1max, edge2max) / min(edge1max, edge2max)
                armor_aspect_ratio = rect_width / rect_height
                armor_area = rect_width * rect_height
                armor_pixel_val = src[int(y), int(x)] ###

                if self._enable_debug:
                    print("*******************************")
                    print("light_angle_diff_:", abs(rect1_angle - rect2_angle))
                    print("radio:", radio)
                    print("armor_angle_:", abs(center_angle))
                    print("armor_aspect_ratio_:", armor_aspect_ratio)
                    print("armor_area_:", armor_area)
                    print("armor_pixel_val_:", src[int(y), int(x)])
                    print("pixel_y", y)
                    print("pixel_x", x)

                angle_diff = abs(rect1_angle - rect2_angle)
                if angle_diff > 175:
                    angle_diff = 180 - angle_diff

                if (angle_diff < self._light_max_angle_diff) and (radio < 2.0) and (armor_aspect_ratio < self._armor_max_aspect_ratio) and (armor_area > self._armor_min_area) and (max(armor_pixel_val) < self._armor_max_pixel_val): ###
                #   cv.putText(src, "light_max_anle_diff: "+str(angle_diff), (int(rect[0][0])-60, int(rect[0][1])-80), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                #    cv.putText(src, "radio: " + str(radio), (int(rect[0][0]-60), int(rect[0][1])-60), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                #    cv.putText(src, "radio: " + str(radio), (int(rect[0][0]-60), int(rect[0][1])-40), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                #    cv.putText(src, "armor_min_area: " + str(armor_area), (int(rect[0][0])-60, int(rect[0][1])-20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1) 
                    if rect1[0][0] < rect2[0][0]:
                        armor_points = self.calcArmorInfo(rect1, rect2)
                        armors.append(Armor(rect, armor_points))
                        
                        if self._enable_debug:
                           DrawRotatedRect(src, rect, (255, 0, 0))
                           extract_rotated_rect(rect, src)
                    else:
                        armor_points = self.calcArmorInfo(rect2, rect1)
                        armors.append(Armor(rect, armor_points))
                        if self._enable_debug:
                           DrawRotatedRect(src, rect, (255, 0, 0))
                           extract_rotated_rect(rect, src)
                           '''
                elif (angle_diff >= self._light_max_angle_diff):
                    cv.putText(src, "light_max_anle_diff: "+str(angle_diff), (int(rect[0][0])-60, int(rect[0][1])-80), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                elif radio >= 2.0:
                    cv.putText(src, "radio: " + str(radio), (int(rect[0][0])-60, int(rect[0][1])-60), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                elif armor_aspect_ratio < self._armor_max_aspect_ratio:
                    cv.putText(src, "armor_aspect_ratio: "+str(armor_aspect_ratio), (int(rect[0][0])-60, int(rect[0][1])-40), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                elif armor_area < self._armor_min_area: 
                    cv.putText(src, "armor_min_area: " + str(armor_area), (int(rect[0][0])-60, int(rect[0][1])-20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    '''

                if self._enable_debug:
                    cv.imshow('armors', src)
        return armors

class FilterArmors:
    def __init__(self, armor_max_stddev, armor_max_mean, enable_debug=False):
        self._armor_max_stddev = armor_max_stddev
        self._armor_max_mean = armor_max_mean
        self._enable_debug = enable_debug

    def __call__(self, armors, src,classifier):
        src = src[:]
        filtered_armors = []
        mask = np.zeros_like(src, cv.CV_8UC1)
        for armor in armors:
            pts = []
            for i in range(4):
                pts.append(armor.vertex[i])
            cv.fillConvexPoly(mask, pts, (0, 255, 0))
            (mean, stddev) = cv.meanStdDev(src)
            if stddev <= self._armor_max_stddev and mean <= self._armor_max_mean:
                filtered_armors.append(armor)

        is_armor = [True for i in filtered_armors]
        for i in range(len(filtered_armors)):
            if is_armor[i]:
                for j in range(i + 1, len(filtered_armors)):
                    if is_armor[j]:
                        dx = filtered_armors[i].rect[0][0] - filtered_armors[j].rect[0][0]
                        dy = filtered_armors[i].rect[0][1] - filtered_armors[j].rect[0][1]
                        dis = math.sqrt(dx * dx + dy * dy)
                        if dis < filtered_armors[i].rect[1][0] + filtered_armors[j].rect[1][0]:
                            if filtered_armors[i].rect[2] > filtered_armors[j].rect[2]:
                                is_armor[i] = False
                            else:
                                is_armor[j] = False

        new_filtered_armors = []
        for i in range(len(filtered_armors)):
            if is_armor[i] and classifier(Armor2Bbox(filtered_armor[i].vertex),src):
                new_filtered_armors.append(filtered_armors[i])
                if self._enable_debug:
                    pass #DrawRotatedRect(src, filtered_armors[i].rect)

        if self._enable_debug:
            cv.imshow('armors_filtered', src)

        return new_filtered_armors

class SelectFinalArmor:
    def __init__(self, enable_debug=False):
        self._enable_debug = enable_debug

    def __call__(self, armors, src):
        armors.sort(cmp=armorCmp)
        if self._enable_debug:
            DrawRotatedRect(src, armors[0].rect)
            cv.imshow('final_armor', src)
        return armors[0]

def Armo2Bbox(armor):
    brect = cv.boundingRect(np.array(armor.vertex))
    bbox = [brect[0],brect[1],brect[2]-brect[0],brect[3]-brect[1]]
    return bbox

def pixel2angle(params,x,y):
    field_x,field_y = params['field_range']
    res_x,res_y     = params['resolution']
    center_x        = res_x//2
    center_y        = res_y//2
    # Using normal convention of cartesian coord
    delta_pitch     = -(center_x - x)*float(field_x)/(res_x)
    delta_yaw       = (center_y - y)*float(field_y)/(res_y)
    return delta_pitch, delta_yaw


class CalcControlInfo:
    def __init__(self, armor_points, intrinsic_matrix, distortion_coeffs, enable_debug=False):
        self._armor_points = armor_points
        self._intrinsic_matrix = intrinsic_matrix
        self._distortion_coeffs = distortion_coeffs
        self._enable_debug = enable_debug

    # TODO: Change Solve PnP to be directly estimate from the intrinsic matrix.
    def __call__(self, armor):
        _, rvec, tvec = cv.solvePnP(self._armor_points, armor.vertex, self._intrinsic_matrix, self._distortion_coeffs)
        if self._enable_debug:
            print("rotation vector:", rvec)
            print("translation vector:", tvec)
        return tvec
# A copy of error code
class ErrorCode:
    OK = 0
    Error = 1

#wrapper function from cmp to key for sorting
def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def extract_rotated_rect(rect, frame):
    
    center = [rect[0][0],rect[0][1]]
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = rect[2] * math.pi / 180

    new = np.zeros([640, 480, 3])
    new_center = [320, 240]
    rotation_mat = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    scretch_mat = np.array([[width/480, 0], [0, height/640]])
    trans_mat = rotation_mat.dot(scretch_mat)
    breakpoint()
    for i in range(640):
        for j in range(480):
            pos = np.array([[j - new_center[1]], [i - new_center[0]]])
            pos = trans_mat.dot(pos)
            

            pos = [int(pos[0][0] + center[0]), int(pos[1][0] + center[1])]
            new[i][j] = frame[pos[1]][pos[0]]
            
    cv.imshow("new", new)
    cv.waitKey(0)


''' Hardware support '''
''' API to industrial camera '''

'''
class Camera:
	def __init__(self, exposure=30,DevInfo):
		self.DevInfo = DevInfo
		self.hCamera = 0
		self.cap = None
		self.pFrameBuffer = 0

	def open(self):
		if self.hCamera > 0:
			return True

		# 打开相机
		hCamera = 0
		try:
			hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
		except mvsdk.CameraException as e:
			print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
			return False

		# 获取相机特性描述
		cap = mvsdk.CameraGetCapability(hCamera)

		# 判断是黑白相机还是彩色相机
		monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

		# 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
		if monoCamera:
			mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
		else:
			mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

		# 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
		FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

		# 分配RGB buffer，用来存放ISP输出的图像
		# 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
		pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

		# 相机模式切换成连续采集
		mvsdk.CameraSetTriggerMode(hCamera, 0)

		# 手动曝光，曝光时间30ms
		mvsdk.CameraSetAeState(hCamera, 0)
		mvsdk.CameraSetExposureTime(hCamera, self.exposure * 1000)
        mvsdk.CameraSetFrameSpeed(hCamera,2)

		# 让SDK内部取图线程开始工作
		mvsdk.CameraPlay(hCamera)

		self.hCamera = hCamera
		self.pFrameBuffer = pFrameBuffer
		self.cap = cap
		return True

	def close(self):
		if self.hCamera > 0:
			mvsdk.CameraUnInit(self.hCamera)
			self.hCamera = 0

		mvsdk.CameraAlignFree(self.pFrameBuffer)
		self.pFrameBuffer = 0

	def grab(self):
		# 从相机取一帧图片
		hCamera = self.hCamera
		pFrameBuffer = self.pFrameBuffer
		try:
			pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
			mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
			mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

			# windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
			# linux下直接输出正的，不需要上下翻转
			if platform.system() == "Windows":
				mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
			
			# 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
			# 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
			frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
			frame = np.frombuffer(frame_data, dtype=np.uint8)
			frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
			return frame
		except mvsdk.CameraException as e:
			if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
				print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
			return None

class SimpleCNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3,8)
        self.bn    = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16)
        self.bn2   = nn.BatchNorm2d(16)
        self.pl    = nn.AdaptiveAvgPool2d(1)
        self.out   = nn.Linear(16,2)

    def train_fw(self,batch):
        h = self.conv1(batch)
        h = self.bn(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.pl(h).view((-1,16))
        out = self.out(h)
        return out
    
    def inference(self,img):
        h = torch.Tensor(img,dtype=torch.float32).view(3,-1)
        h = self.conv1(img)
        h = self.bn(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.pl(h).view((-1,16))
        out = self.out(h)
        return out

class CNNClassifier:
    def __init__(self,path):
        self.path = path
        self.model= SampleCNN()
        self.model.load_state_dict(torch.load(path))

    def __call__(self,img,bbox):
        bbox = [[int(bbox[0]),int(bbox[1])],[int(bbox[2]+bbox[0]),int(bbox[3]+bbox[1])]]
        roi  = img[:,bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1]]
        roi  = cv.resize(roi,(32,32))
        ret  = bool(torch.argmax(self.model.inference(roi).view(-1)))
        return ret
'''

        
        
