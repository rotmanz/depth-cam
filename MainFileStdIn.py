import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import configparser
from datetime import datetime
import os.path
import csv
from lib.helpers import *
from lib import calibration, surface_fit_features as sfs, cushions_straight as cs , surface_fit_features_No_Z as sfnz
from lib.point_filters import Filter
from lib.live_capture import LiveCapture
import sys
import time
import shlex


class Settings:
    def __init__(self,setting_file):
        self.config = configparser.ConfigParser()
        self.config.sections()
        try:
            self.config.read(setting_file)
        except():
            raise NameError("cannot open settings file {}".format(setting_file))

        config = self.config

        self.resX = int(config['camera']['resX'])
        self.resY = int(config['camera']['resY'])
        self.frameRate = int(config['camera']['frameRate'])
        self.max_laser = config['camera']['max_laser'] == 'True'

        # [processing]
        self.initIterations = int(config['processing']['initIterations'])
        self.iterations = int(config['processing']['iterations'])
        self.fit_order = int(config['processing']['fit_order'])
        self.order_max_dif = float(config['processing']['order_max_dif'])
        self.max_diff_factor = float(config['processing']['max_diff_factor'])
        self.isBrown = config['processing']['is_brown'] == 'True'
        # boxes_to_shift = np.asarray(
        #     [int(x) for x in config['processing']['boxes_shift'].split(',') if x.strip().isdigit()])

        self.liftCoeffs = np.array(
            [float(config['lift']['coefX']) , float(config['lift']['coefY']) , float(config['lift']['coefZ'])])

        # [capture]
        self.capt_use_live_capture = config['capture']['use_live_capture'] == 'True'
        self.capt_single_capture_attempts = int(config['capture']['single_capture_attempts'])
        self.capt_single_capture_timeout_ms = int(config['capture']['single_capture_timeout_ms'])
        self.capt_align_to = int(config['capture']['align_to'])  # 0 is Depth, 1 to Color
        self.capt_saving_subfolder = config['capture']['saving_subfolder']
        self.isFramos = config['capture']['is_framos'] == 'True'
        self.ROI = np.zeros(4 , dtype=np.int)
        self.ROI[0] = int(config['capture']['ROI_min_x'])
        self.ROI[1] = int(config['capture']['ROI_max_x'])
        self.ROI[2] = int(config['capture']['ROI_min_y'])
        self.ROI[3] = int(config['capture']['ROI_max_y'])
        self.A_factor = float(config['capture']['A_factor'])

        # [calibration]
        self.cal_perform_rotation = config['calibration']['perform_rotation'] == 'True'
        self.cal_perform_dual_rotation = config['calibration']['perform_dual_rotation'] == 'True'
        self.cal_display_angles = config['calibration']['display_angles'] == 'True'
        self.cal_use_ransac = config['calibration']['use_ransac'] == 'True'
        self.cal_use_filter_type = config['calibration']['use_filter_type']
        self.cal_multi_image = int(config['calibration']['multi_image'])  # 0 or -X are single image

        # External options - [default]
        self.save_folder_name = config['default']['save_folder_name']
        self.fromFile = config['default']['fromBag'] == 'True'
        self.fromNpz = config['default']['fromNpz'] == 'True'
        self.save_line = config['default']['save_line'] == 'True'


# def findMedia(c , colorImage):
#     stacked_img = np.stack((c ,) * 3 , axis=-1).astype(np.uint8)
#     mask = np.zeros(stacked_img.shape[:2] , np.uint8)
#
#     bgdModel = np.zeros((1 , 65) , np.float64)
#     fgdModel = np.zeros((1 , 65) , np.float64)
#
#     rect = (0 , 10 , 640 , 470)
#     cv2.grabCut(stacked_img , mask , rect , bgdModel , fgdModel , 5 , cv2.GC_INIT_WITH_RECT)
#
#     tempImage = np.std(colorImage , axis=2) / np.mean(colorImage , axis=2)
#     tempImage[np.mean(colorImage , axis=2) == 0] = 255
#     mask_from_color = np.zeros(stacked_img.shape[:2] , np.uint8)
#     mask_from_color[tempImage < 0.2] = 1
#     kernel = np.ones((5 , 5) , np.uint8)
#     mask_from_color = cv2.morphologyEx(mask_from_color , cv2.MORPH_OPEN , kernel)
#
#     mask[120:170 , 400:450] = 1
#     mask[mask_from_color == 1] = 1
#     mask[70:270 , 600:639] = 0
#     mask[:10 , :] = 0
#     mask , bgdModel , fgdModel = cv2.grabCut(stacked_img , mask , None , bgdModel , fgdModel , 5 ,
#                                              cv2.GC_INIT_WITH_MASK)
#
#     mask2 = np.where((mask == 2) | (mask == 0) , 0 , 1).astype('uint8')
#     return mask2

def writeOutput(IF,boxes,Z_mean_min_max,w):
    OF = IF.replace("input","output")

    # change units from meters to mm
    w = w * 1000

    if OF==IF:
        fname,ext = os.path.splitext(OF)
        OF = fname+'_output'+ext

    Z_mean_min_max = np.max(Z_mean_min_max)-Z_mean_min_max
    i = [0 , 2 , 1]
    Z_mean_min_max = Z_mean_min_max[: , i]

    with open(OF , 'w' , newline='\n') as csvfile:
        ofwriter = csv.writer(csvfile , delimiter=',')
        temprow=list()
        temprow.append(boxes[0][0])
        ofwriter.writerow(temprow+['AVG','Min','Max'])
        for box in range(len(Z_mean_min_max)):
            temprow = list()
            temprow.append(boxes[box+1][0])
            ofwriter.writerow(temprow+list((str(ele*1000) for ele in Z_mean_min_max[box])))
        ofwriter.writerow([])
        ofwriter.writerow(['Warp']+[str(w)])

def save_data_to_PC(PCF, averageImage , irImage , colorImage , x , y , z , mask2 , plane_coefficients , boxes  ):
    now = datetime.now()
    outfile = PCF + '\\' + now.strftime("cloud_calc_%d%m%Y_%H%M%S")
    np.savez(outfile , averageImage , irImage , colorImage , x , y , z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-FF", help="reads image stream from file instead of camera")
    parser.add_argument("-SF", help="settings file used for capture and process stages. settings are being modified by calibraion process")
    parser.add_argument("-IF", help="input file specifies measurement locations and media dimension")
    parser.add_argument("-PCF", help="specifies printcare folder for application telemetry")
    parser.add_argument("-OP", help="specifies the operation supported values: BIT Capture and Calibrate")
    parser.add_argument("-VF", help="this file will be used during BIT only to save version name")
    # args = parser.parse_args()

    pipeStarted=False
    live_capture = None
    mask2 = None

    settings = None



    while True:
        in_line = sys.stdin.readline()
        in_line_list = shlex.split(in_line)
        args = parser.parse_args(in_line_list)

        Exited = False

        if args.OP:
            if args.OP=='BIT':
                ExitedOnce = False
                print('BIT selected',flush=True)
                if args.SF is None:
                    # sys.exit(-2)
                    print('\nExit,-10\n ',flush=True)
                    Exited = True
                try:
                    settings = Settings(args.SF)
                except():
                    # sys.exit(-2)
                    print('\nExit,-10\n ',flush=True)
                    Exited = True

                if not pipeStarted:
                    live_capture = LiveCapture(align=settings.capt_align_to,is_framos=settings.isFramos,ROI = settings.ROI,A_factor=settings.A_factor,max_laser=settings.max_laser)

                if args.VF:
                    if os.path.isfile(args.VF):
                        f = open(args.VF,"w")
                        f.write('1.0.5')
                        f.close()
                    else:
                        print("Version file does not exist",flush=True)
                try:
                    live_capture.start()
                    pipeStarted = True
                    data_dict = live_capture.capture_average_reading(dry_runs=1 ,
                                                             images_2_mean_count=1 ,
                                                             mean_flag=False ,
                                                             single_capture_attempts=1 ,
                                                             single_capture_timeout_ms=100)
                    # live_capture.stop()
                except ConnectionError as e:
                    print(e,flush=True)
                    # live_capture.pipe = None
                    # sys.exit(-1)
                    # print('\nExit,-1\n ',flush=True)
                    print('first attempt failed\n ' , flush=True)
                    ExitedOnce = True

                if ExitedOnce:
                    live_capture.reset()
                    try:
                        live_capture.start()
                        pipeStarted = True
                        data_dict = live_capture.capture_average_reading(dry_runs=1 ,
                                                                         images_2_mean_count=1 ,
                                                                         mean_flag=False ,
                                                                         single_capture_attempts=1 ,
                                                                         single_capture_timeout_ms=100)
                        # live_capture.stop()
                    except ConnectionError as e:
                        print(e , flush=True)
                        live_capture.pipe = None
                        # sys.exit(-1)
                        print('second attempt failed\n ' , flush=True)
                        print('\nExit,-1\n ',flush=True)

                        Exited = True
                if not Exited:
                    print('\nExit,0\n ',flush=True)


            elif args.OP=='Capture':
                print('Capture selected',flush=True)
                start = time.time()
                if args.SF is None or args.IF is None:
                    # sys.exit(-2)
                    print('\nExit,-2\n ',flush=True)
                    Exited = True
                if not os.path.isfile(args.SF) or not os.path.isfile(args.IF):
                    # sys.exit(-2)
                    print('\nExit,-2\n ',flush=True)
                    Exited = True
                try:
                    settings = Settings(args.SF)
                except():
                    # sys.exit(-2)
                    print('\nExit,-2\n ',flush=True)
                    Exited = True

                if args.FF:
                    f = np.load(args.FF)
                    averageImage = f['arr_0']  # averageImage
                    irImage = f['arr_1']  # self.c
                    colorImage = f['arr_2']  # colorImage
                    x = f['arr_3']
                    y = f['arr_4']
                    z = f['arr_5']
                else:

                    if not pipeStarted:
                        live_capture = LiveCapture(align=settings.capt_align_to,is_framos=settings.isFramos,ROI = settings.ROI,A_factor=settings.A_factor,max_laser=settings.max_laser)
                        pipeStarted = True
                        try:
                            live_capture.start()
                        except ConnectionError as e:
                            print(e)
                            live_capture.pipe = None
                            # sys.exit(-20)
                            print('\nExit,-20\n ',flush=True)
                            Exited = True

                    try:
                        capture_start = time.time()
                        data_dict = live_capture.capture_average_reading(dry_runs=settings.initIterations ,
                                                                      images_2_mean_count=settings.iterations ,
                                                                      mean_flag=False ,
                                                                      single_capture_attempts=settings.capt_single_capture_attempts ,
                                                                      single_capture_timeout_ms=settings.capt_single_capture_timeout_ms)

                        x , y , z = data_dict['X'] , data_dict['Y'] , data_dict['Z']
                        irImage, colorImage , averageImage = data_dict['ir'] , data_dict['color'] , data_dict['depth_raw']
                    except:
                        print("image grabbing failed",flush=True)
                        live_capture.pipe = None
                        # sys.exit(-200)
                        print('\nExit,-200\n ',flush=True)
                        Exited = True

                    # capt.stop()


                if not Exited:

                    data_load_end = time.time()

                    with open(args.IF , newline='') as csvfile:
                        boxes = list(csv.reader(csvfile))

                    seperator = -1

                    for line in range(len(boxes)):
                        if all(ele == '' for ele in boxes[line]):
                            seperator = line
                            break

                    if seperator > 1:
                        mediaWidth = int(boxes[seperator + 1][1])
                        settings.isBrown = boxes[seperator + 2][1] == 'b'


                    if mask2 is None:
                        mask2 = findMedia(irImage,colorImage, isBrown=settings.isBrown , zImage=z)

                    if np.sum(mask2==1) > 1:

                        find_media_end = time.time()

                        cal = calibration.Calibration()
                        cal.plane_coefficients = settings.liftCoeffs

                        # with open(args.IF , newline='') as csvfile:
                        #     boxes = list(csv.reader(csvfile))
                        #
                        # seperator = -1
                        #
                        # for line in range(len(boxes)):
                        #     if all(ele == '' for ele in boxes[line]):
                        #         seperator = line
                        #         break
                        #
                        # if seperator>1:
                        #     mediaWidth = int(boxes[seperator+1][1])

                        boxesCoordinates = np.asarray(boxes[1:seperator])
                        boxesCoordinates = boxesCoordinates[: , 1:5].astype(np.int)

                        boxesCoordinates = boxesCoordinates[:,[2,3,0,1]] #change print and cross-print input order

                        boxesMax = np.asarray(boxes[1:seperator])
                        boxesMax = boxesMax[: , 5].astype(np.int)

                        point_array = np.vstack((x[mask2 == 1].flatten() , y[mask2 == 1].flatten() ,
                                                 z[mask2 == 1].flatten())).transpose()
                        point_array = point_array[point_array[: , 2] > 0 , :]

                        rotatedPoints = cal.rotate_about_calibration(point_array ,
                                                                          coefficients=None ,
                                                                          with_print_log=settings.cal_display_angles ,
                                                                          dual_rotation=settings.cal_perform_dual_rotation)

                        rotation_end = time.time()

                        rotatedPoints = rotatedPoints.astype(np.float32)

                        rotatedPoints[:,0] = np.max(rotatedPoints[:,0]) - rotatedPoints[:,0] + 0.05

                        # csdata , C , X , Y , Z , r2 = sfs.fit_clean_surf(rotatedPoints , order=3 , iter=5 , coeff_mx_change=0.05)
                        # csdata , C , X , Y , Z , r2 , s = cs.fit_clean_surf(rotatedPoints , order=3 , iter=5 , coeff_mx_change=0.05)
                        csdata , C , res = sfnz.fit_clean_surf(rotatedPoints , order=3 , iter=5 , coeff_mx_change=0.01)

                        csdata[:,1] = csdata[:,1] - np.min(csdata[:,1])

                        boxes_csdata = csdata[csdata[:,1]<(np.max(boxesCoordinates[:,1]/1000+0.1)),:]


                        polyPoints = np.array([[np.min(boxes_csdata[: , 0]) , np.min(boxes_csdata[: , 1])] ,
                                               [np.max(boxes_csdata[: , 0]) , np.max(boxes_csdata[: , 1])]])
                        # C , X , Y , Z , order = sfs.parallel_fit_auto_order(csdata , selected_point_list=polyPoints, max_diff=settings.order_max_dif)
                        # C , X , Y , Z , order = sfs.fit_auto_order(csdata , selected_point_list=polyPoints , board_height = mediaWidth,
                        #                                            max_diff=settings.order_max_dif,max_diff_factor=settings.max_diff_factor)

                        C , order = sfnz.fit_auto_order(boxes_csdata , max_diff=settings.order_max_dif ,
                                                        max_diff_factor=settings.max_diff_factor)

                        print("order selected: {}".format(order),flush=True) # Ziv 055020 added print of order for debug purposes (will be saved to STA log)

                        # w , warp_type , frameMinIsGlobalMin , frameMaxIsGlobalMax = sfs.board_warp(Z , d1row=4 , d2row=205 ,
                        #                                                                            d1col=4 , d2col=205)
                        # Z_mean_min_max = sfs.fit_Z_to_rectangles(Z , boxesCoordinates , boxesMax)

                        w , warp_type , frameMinIsGlobalMin , frameMaxIsGlobalMax = sfnz.board_warp(boxes_csdata , C , d1row=4 ,
                                                                                                    d2row=205 , d1col=4 ,
                                                                                                    d2col=205 ,
                                                                                                    selected_point_list=polyPoints ,
                                                                                                    res_X_Y=0.005)
                        Z_mean_min_max = sfnz.fit_Z_to_rectangles(boxes_csdata , C , boxesCoordinates , avg_high_perc=boxesMax ,
                                                                  selected_point_list=polyPoints , board_height=mediaWidth , X=None ,
                                                                  Y=None , res_X_Y=0.001)

                        surface_processing_end = time.time()

                        writeOutput(args.IF,boxes,Z_mean_min_max,w)

                        end = time.time()
                        if not args.FF:
                            print("time from capture start: {}".format(end - capture_start),flush=True)
                        print("time from load: {}".format(end - data_load_end),flush=True)
                        print("time from find media: {}".format(end - find_media_end),flush=True)
                        print("time from rotation: {}".format(end - rotation_end),flush=True)
                        print("time from surface processing: {}".format(end - surface_processing_end),flush=True)
                        print("time elapsed: {}".format(end-start),flush=True)

                        if args.PCF:
                            save_data_to_PC(args.PCF, averageImage , irImage , colorImage , x , y , z , mask2 , cal.plane_coefficients , boxesCoordinates  )

                        print('\nExit,0\n ',flush=True)
                    else:
                        print('\nExit,-220\n ' , flush=True)

            elif args.OP == 'Calibrate':
                print('Calibrate selected',flush=True)
                if args.SF is None or args.IF is None:
                    # sys.exit(-3)
                    print('\nExit,-3\n ',flush=True)
                if not os.path.isfile(args.SF) or not os.path.isfile(args.IF):
                    # sys.exit(-3)
                    print('\nExit,-3\n ',flush=True)
                try:
                    settings = Settings(args.SF)
                except():
                    # sys.exit(-3)
                    print('\nExit,-3\n ',flush=True)

                # config = configparser.ConfigParser()
                # config.sections()
                # try:
                #     config.read(args.SF)
                # except():
                #     sys.exit(-3)
                #
                # # [camera]
                # resX = int(config['camera']['resX'])
                # resY = int(config['camera']['resY'])
                # frameRate = int(config['camera']['frameRate'])
                # # [processing]
                # initIterations = int(config['processing']['initIterations'])
                # iterations = int(config['processing']['iterations'])
                # fit_order = int(config['processing']['fit_order'])
                # # boxes_to_shift = np.asarray(
                # #     [int(x) for x in config['processing']['boxes_shift'].split(',') if x.strip().isdigit()])
                #
                # liftCoeffs = np.array(
                #     [float(config['lift']['coefX']) , float(config['lift']['coefY']) , float(config['lift']['coefZ'])])
                #
                # # [capture]
                # capt_use_live_capture = config['capture']['use_live_capture'] == 'True'
                # capt_single_capture_attempts = int(config['capture']['single_capture_attempts'])
                # capt_single_capture_timeout_ms = int(config['capture']['single_capture_timeout_ms'])
                # capt_align_to = int(config['capture']['align_to'])  # 0 is Depth, 1 to Color
                # capt_saving_subfolder = config['capture']['saving_subfolder']
                #
                # # [calibration]
                # cal_perform_rotation = config['calibration']['perform_rotation'] == 'True'
                # cal_perform_dual_rotation = config['calibration']['perform_dual_rotation'] == 'True'
                # cal_display_angles = config['calibration']['display_angles'] == 'True'
                # cal_use_ransac = config['calibration']['use_ransac'] == 'True'
                # cal_use_filter_type = config['calibration']['use_filter_type']
                # cal_multi_image = int(config['calibration']['multi_image'])  # 0 or -X are single image
                #
                # # External options - [default]
                # save_folder_name = config['default']['save_folder_name']
                # fromFile = config['default']['fromBag'] == 'True'
                # fromNpz = config['default']['fromNpz'] == 'True'
                # save_line = config['default']['save_line'] == 'True'

                capt = LiveCapture(align=settings.capt_align_to,is_framos=settings.isFramos,A_factor=settings.A_factor)
                try:
                    capt.start()
                except ConnectionError as e:
                    print(e,flush=True)
                    capt.pipe = None

                data_dict = capt.capture_average_reading(dry_runs=settings.initIterations ,
                                                         images_2_mean_count=settings.iterations ,
                                                         mean_flag=False ,
                                                         single_capture_attempts=settings.capt_single_capture_attempts ,
                                                         single_capture_timeout_ms=settings.capt_single_capture_timeout_ms)
                x , y , z = data_dict['X'] , data_dict['Y'] , data_dict['Z']
                irImage , colorImage , averageImage = data_dict['ir'] , data_dict['color'] , data_dict['depth_raw']

                cal = calibration.Calibration()

                # mask2 = findMedia(self.c , colorImage , isBrown=self.isBrown , zImage=self.z)
                mask2 = findMedia(irImage , colorImage, isBrown=settings.isBrown , zImage=z)

                point_array = np.vstack((x[mask2 == 1].flatten() , y[mask2 == 1].flatten() ,
                                         z[mask2 == 1].flatten())).transpose()
                if settings.cal_multi_image > 0:
                    # init memory
                    coef_arr = np.zeros([settings.cal_multi_image , 3])
                    # main capturing loop
                    for i in range(settings.cal_multi_image):

                        data_dict = capt.capture_average_reading(dry_runs=initIterations ,
                                                                 images_2_mean_count=iterations ,
                                                                 mean_flag=False ,
                                                                 single_capture_attempts=capt_single_capture_attempts ,
                                                                 single_capture_timeout_ms=capt_single_capture_timeout_ms)
                        x , y , z = data_dict['X'] , data_dict['Y'] , data_dict['Z']
                        irImage , colorImage , averageImage = data_dict['ir'] , data_dict['color'] , data_dict['depth_raw']

                        point_array = np.vstack((x[mask2 == 1].flatten() , y[mask2 == 1].flatten() ,
                                                 z[mask2 == 1].flatten())).transpose()
                        # Filtering switch
                        if cal_use_filter_type == 'box':
                            filt = Filter()
                            filt.point_array = point_array
                            filt.box((-0.9 , -0.9 , 0.6) , (1.8 , 1.8 , 1.0))
                            self.cal.clean_point_array = filt.point_array
                        elif cal_use_filter_type == 'sphere':
                            raise NotImplementedError("Spherical filter is not implemented in caputeUI")
                            cal.clean_point_array = point_array
                        elif cal_use_filter_type == 'None':
                            cal.clean_point_array = point_array
                        else:
                            csdata , C , X , Y , Z , r2 , s = cs.fit_clean_surf(point_array , order=3 , iter=5 ,
                                                                                coeff_mx_change=0.01)
                            self.cal.clean_point_array = csdata
                        # Plane finder
                        cal.find_plane_coefficients(use_ransac=self.cal_use_ransac)
                        coef_arr[i , :] = np.asarray(self.cal.plane_coefficients)  # store last coeffs
                    cal.plane_coefficients = tuple(np.mean(coef_arr , axis=0))

                else:
                    # Filtering switch
                    if settings.cal_use_filter_type == 'box':
                        filt = Filter()
                        filt.point_array = point_array
                        filt.box((-0.9 , -0.9 , 0.6) , (1.8 , 1.8 , 1.0))
                        cal.clean_point_array = filt.point_array
                    elif settings.cal_use_filter_type == 'sphere':
                        raise NotImplementedError("Spherical filter is not implemented in caputeUI")
                        cal.clean_point_array = point_array
                    elif settings.cal_use_filter_type == 'None':
                        cal.clean_point_array = point_array
                    else:
                        csdata , C , X , Y , Z , r2 , s = cs.fit_clean_surf(point_array , order=3 , iter=5 ,
                                                                            coeff_mx_change=0.01)
                        cal.clean_point_array = csdata
                    self.cal.find_plane_coefficients(use_ransac=self.cal_use_ransac)

                config['lift']['coefX'] = str(cal.plane_coefficients[0])
                config['lift']['coefY'] = str(cal.plane_coefficients[1])
                config['lift']['coefZ'] = str(cal.plane_coefficients[2])
                with open(args.SF , 'w') as configfile:
                    config.write(configfile)
                print('\nExit,0\n ',flush=True)

            elif args.OP == 'Quit':
                if pipeStarted:
                    live_capture.stop()
                sys.exit(0)





            else:
                print (args.OP + ' is not a supported operation',flush=True)
                print('\nExit,-6\n ',flush=True)








if __name__ == '__main__':
    main()





