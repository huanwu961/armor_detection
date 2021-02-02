import cv2
import numpy as np
import roborts_detection_utils as utils
import json
import time

def crop():
    
    path = "./icra/icra_data/armor_data_1/data0.png"

    cv2.namedWindow("trackBar")
    cv2.createTrackbar("HueMin", "trackBar", 0, 255, callable)
    cv2.createTrackbar("HueMax", "trackBar", 0, 255, callable)
    cv2.createTrackbar("SurMax", "trackBar", 0, 255, callable)
    cv2.createTrackbar("SurMin", "trackBar", 0, 255, callable)
    cv2.createTrackbar("ValMax", "trackBar", 0, 255, callable)
    cv2.createTrackbar("ValMin", "trackBar", 0, 255, callable)

    
    #if (debug == True):
        #processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    k = 0
    while True:
        frame = cv2.imread(path)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        HueMax = cv2.getTrackbarPos("HueMax", "trackBar")
        SurMax = cv2.getTrackbarPos("SurMax", "trackBar")
        ValMax = cv2.getTrackbarPos("ValMax", "trackBar")
        HueMin = cv2.getTrackbarPos("HueMin", "trackBar")
        SurMin = cv2.getTrackbarPos("SurMin", "trackBar")
        ValMin = cv2.getTrackbarPos("ValMin", "trackBar")

        #print((HueMin, SurMin, ValMin, HueMax, SurMax, ValMax))

        mask = cv2.inRange(processed_frame, (83, 0, 128), (255, 255, 255), None)
        #frame = np.array(frame)
        #mask = np.array(mask)
        new_frame = cv2.bitwise_and(frame, frame, mask=mask)

        contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #cv2.imshow("test", new_frame)
        cv2.waitKey(1)

class test():
    def __init__(self, frame):
        self.frame = frame
        with open('detector_params.json', 'r') as file:
            data = json.load(file)
            print(type(data))
        #string = f.read()
        #print(string)
        #j_string = json.dumps(string)

        #self.params = json.loads(string)
        debug = True
        self.params = data
        print(type(self.params))
        self.det_thresholdcolor = utils.BGRImageProc(color=self.params['color'],
                                                     threshs=self.params['bgr_threshs'],
                                                     enable_debug=debug)
        self.det_gray = utils.GrayImageProc()

        self.det_getlightbars   = utils.ScreenLightBars(thresh=self.params['bright_thresh'],
                                                        enable_debug=debug)

        self.det_filterlightbars= utils.FilterLightBars(light_max_aspect_ratio=self.params['light_max_aspect_ratio'],
                                                        light_min_area=self.params['light_min_area'],
                                                        enable_debug=debug)

        self.det_possiblearmors = utils.PossibleArmors(light_max_angle_diff=self.params['light_max_angle_diff'],
                                                       armor_max_aspect_ratio=self.params['armor_max_aspect_ratio'],
                                                       armor_min_area=self.params['armor_min_area'],
                                                       armor_max_pixel_val=self.params['armor_max_pixel_val'],
                                                       enable_debug=debug)

        '''self.det_filterarmors = utils.FilterArmors(armor_max_stddev=self.params['armor_max_stddev'],
                                                   armor_max_mean  =self.params['armor_max_mean'],
                                                   enable_debug=debug)

        self.det_selectarmor  = utils.SelectFinalArmor(enable_debug=debug)

        self.det_armor2box = utils.Armo2Bbox()
        '''
    
    def test_run(self):
        
        gray            = self.det_gray(self.frame)
        #cv2.imshow(gray)

        thresh          = self.det_thresholdcolor(self.frame)
        cv2.imshow("thresh",thresh)

        lightbars, img       = self.det_getlightbars(thresh,gray,self.frame)
        cv2.imshow("lightBars", img)
        rects           = self.det_filterlightbars(lightbars,self.frame)

        armors          = self.det_possiblearmors(rects,self.frame)
#        cv2.rectangle()
#        armors          = self.det_filterarmors(armors,frame,classifier)
#        target          = self.det_selectarmor(armors,frame)
#        bbox            = self.det_armor2box(target)

def main():
    realtime = False 
    path = "./icra_data/armor_data_1/data1.png"
    #frame = cv2.imread(path)
    if realtime:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("cannot open the camera!")
            exit(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("cannot reiceve any picture from the camera..")
                exit(1)
            test1 = test(frame)
            test1.test_run()
            if cv2.waitKey(1) == ord('q'):
                break
    else:
        for i in range(374):
            if (i == 9):
                i+=1
            path = "./icra_data/armor_data_1/data" + str(i) + ".png"
            frame = cv2.imread(path)
            test1 = test(frame)
            test1.test_run()
            cv2.waitKey(250)
        
    #cv2.waitKey(0)

main()

