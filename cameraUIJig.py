try:
    import tkinter as tk  # for python 3
except:
    import Tkinter as tk  # for python 2

import pygubu
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
# Implement the default Matplotlib key bindings.
import configparser
from datetime import datetime
import os.path
import csv
from tkinter import filedialog

# Project classes and files
from lib.line import LineExtractor
from lib.helpers import *
from lib import calibration, surface_fit_features as sfs, cushions_straight as cs, surface_fit_features_No_Z as sfnz
from lib.point_filters import Filter
from lib.live_capture_jig import LiveCapture
# from lib.live_capture_framos import LiveCapture

#
# def convert_depth_frame_to_pointcloud(depth_image , camera_intrinsics, depth_scale):
#     """
#     Convert the depthmap to a 3D point cloud
#     Parameters:
#     -----------
#     depth_frame 	 	 : rs.frame()
#                            The depth_frame containing the depth map
#     camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
#     Return:
#     ----------
#     x : array
#         The x values of the pointcloud in meters
#     y : array
#         The y values of the pointcloud in meters
#     z : array
#         The z values of the pointcloud in meters
#     """
#
#     [height , width] = depth_image.shape
#
#     nx = np.linspace(0 , width - 1 , width)
#     ny = np.linspace(0 , height - 1 , height)
#     u , v = np.meshgrid(nx , ny)
#     x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
#     y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy
#
#     z = depth_image.flatten() * depth_scale
#     x = np.multiply(x , z)
#     y = np.multiply(y , z)
#
#     # x = x[np.nonzero(z)]
#     # y = y[np.nonzero(z)]
#     # z = z[np.nonzero(z)]
#
#     x = x.reshape(depth_image.shape)
#     y = y.reshape(depth_image.shape)
#     z = z.reshape(depth_image.shape)
#
#     return x , y , z


class Application:
    def __init__(self, master, config_file='config.ini'):

        #1: Create a builder
        self.builder = builder = pygubu.Builder()

        #2: Load an ui file
        builder.add_from_file('cameraUI.ui')

        #3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('mainwindow', master)
        #  connect buttons to functions
        builder.connect_callbacks(self)

        # UI elements needed for redrawing
        self.canvas = None
        self.fig = None
        # user selected media edge
        self.poly = np.array([[], []], dtype=np.int32).transpose()
        # ir image used for selecting points
        self.c = None

        self.cal = calibration.Calibration()

        if os.path.exists('generic_coef.json'):
            self.cal.load_plane_info('generic_coef.json' , method='json')
        else:
            tk.messagebox.showwarning(title='Missing Calibration' , message='No stored calibration file please run calibration')

        with open('boxes.csv', newline='') as csvfile:
            self.boxes = np.asarray(list(csv.reader(csvfile)), dtype=np.int)

        # !!!!! DEPRACATED, added by Elad as an attempt to set the boxes in the image !!!!!!!!!!!
        # box = self.boxes
        # import matplotlib.patches as patches
        # self.rects = []
        # for i in range(self.boxes.shape[0]):
        #     rect_dim = [(box[i,0] , box[i,2]), abs(box[i,1] - box[i,0]), abs(box[i,3] - box[i,2])]
        #     self.rects.append(patches.Rectangle(rect_dim[0], rect_dim[1], rect_dim[2], linewidth=1, edgecolor='c', facecolor='none'))

        config = configparser.ConfigParser()
        config.sections()
        try:
            config.read(config_file)
        except():
            raise RuntimeError(f"Problem with reading config file: \"{config_file}\".")

        # [camera]
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
        self.boxes_to_shift = np.asarray([int(x) for x in config['processing']['boxes_shift'].split(',') if x.strip().isdigit()])
        self.isBrown = config['processing']['is_brown'] == 'True'

        # self.liftCoeffs = np.array([float(config['lift']['coefX']),float(config['lift']['coefY']),-1])

        # [capture]
        self.capt_use_live_capture = config['capture']['use_live_capture'] == 'True'
        self.capt_single_capture_attempts = int(config['capture']['single_capture_attempts'])
        self.capt_single_capture_timeout_ms = int(config['capture']['single_capture_timeout_ms'])
        self.capt_align_to = int(config['capture']['align_to']) # 0 is Depth, 1 to Color
        self.capt_saving_subfolder = config['capture']['saving_subfolder']
        self.isFramos = config['capture']['is_framos'] == 'True'
        self.ROI = np.zeros(4,dtype = np.int)
        self.ROI[0] = int(config['capture']['ROI_min_x'])
        self.ROI[1] = int(config['capture']['ROI_max_x'])
        self.ROI[2] = int(config['capture']['ROI_min_y'])
        self.ROI[3] = int(config['capture']['ROI_max_y'])
        self.A_factor = float(config['capture']['A_factor'])
        self.exposure = float(config['capture']['exposure'])

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
        # Opening a camera pipeline
        if self.capt_use_live_capture:
            self.capt = LiveCapture(align=self.capt_align_to,is_framos=self.isFramos,ROI = self.ROI,A_factor=self.A_factor,max_laser=self.max_laser,exposure=self.exposure)
            try:
                self.capt.start()
            except ConnectionError as e:
                print(e)
                self.capt.pipe = None

        self.mask2 = None

    def clearPoly(self):
        # used to clear user selected points and redraw image
        self.poly=np.array([[],[]] , dtype=np.int32).transpose()
        if self.c is not None:
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            self.ax.imshow(self.c)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()

    def loadData(self):
        file_path = filedialog.askopenfilename()
        f = np.load(file_path)
        averageImage = f['arr_0']  # averageImage
        self.c = f['arr_1']  # self.c
        colorImage = f['arr_2']  # colorImage
        self.x = f['arr_3']
        self.y = f['arr_4']
        self.z = f['arr_5']

        self.redrawCanvas()

        # self.mask2 = findMedia(self.c,colorImage, isBrown=self.isBrown,zImage=self.z)

        self.mask2 = np.ones(self.c.shape)

        overlay = np.copy(self.c) * self.mask2
        alpha = 0.8
        image_new = cv2.addWeighted(overlay , alpha , self.c , 1 - alpha , 0)
        self.ax.imshow(image_new)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()







        # if self.fig is None:
        #     self.fig = plt.figure()
        #     self.fig.clf()
        #     self.ax = self.fig.add_subplot(111)
        # else:
        #     self.clearPoly()
        #
        # self.ax.imshow(self.c)
        #
        # imageCanvas = self.builder.get_object('imageCanvas', self.mainwindow)
        # imageCanvas.delete("all")
        #
        # if self.canvas is None:
        #     self.canvas = FigureCanvasTkAgg(self.fig , master=imageCanvas)  # A tk.DrawingArea.
        # self.canvas.draw()
        # self.canvas.get_tk_widget().pack()
        # self.canvas.mpl_connect('button_press_event' , self.on_key_press)


    def capture(self, loadNPZ=False, load_filename=""):
        # method used to capture and save data from depth camera
        # it also displays IR image for media edge selection
        time_start = datetime.now()
        if self.capt_use_live_capture:
            if self.capt.pipe is not None:
                data_dict = self.capt.capture_average_reading(dry_runs=self.initIterations,
                                                              images_2_mean_count=self.iterations,
                                                              mean_flag=False,
                                                              single_capture_attempts=self.capt_single_capture_attempts,
                                                              single_capture_timeout_ms=self.capt_single_capture_timeout_ms)
                self.x, self.y, self.z = data_dict['X'], data_dict['Y'], data_dict['Z']
                self.c, colorImage, averageImage = data_dict['ir'], data_dict['color'], data_dict['depth_raw']
            else:
                print("Device can not be resolved")
                return
        else:
            # original start
            pipeline = rs.pipeline()
            cfg = rs.config()

            if self.fromFile:
                cfg.enable_device_from_file(
                    r"C:\Temp\realsense\lp2\2002\2100x1300\take 3\20200220_112327_laser without cognx.bag")
            else:
                # cfg.enable_stream(rs.stream.depth , self.resX , self.resY , rs.format.z16 , self.frameRate)
                # cfg.enable_stream(rs.stream.color , self.resX , self.resY , rs.format.bgr8 , self.frameRate)
                # cfg.enable_stream(rs.stream.infrared , 1 , self.resX , self.resY , rs.format.y8 , self.frameRate)
                # cfg.enable_stream(rs.stream.infrared , 2 , self.resX , self.resY , rs.format.y8 , self.frameRate)
                cfg.enable_all_streams()

            profile = pipeline.start(cfg)

            if not self.fromFile:
                device = profile.get_device()
                depth_sensor = device.query_sensors()[0]
                laser_range = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power , laser_range.max)

            try:
                for imageNum in range(self.initIterations):
                    # Wait for a coherent pair of frames: depth and color
                    frames = pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
    # TODO - This is received from the *.ini file - it is overwritten here - problem needs solution
                resX = depth_frame.width
                resY = depth_frame.height

                images = np.zeros((resY , resX , self.iterations))
                # TODO - if this part not depracated add a call to alignemt from *.ini file
                align_to = rs.stream.depth
                align = rs.align(align_to)

                for imageNum in range(self.iterations):
                    # Wait for a coherent pair of frames: depth and color
                    frames = pipeline.wait_for_frames()

                    aligned_frames = align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    ir1_frame = aligned_frames.get_infrared_frame(1) # 1 is the right-hand cam (far from RGB)

                    depth_image = np.asanyarray(depth_frame.get_data())

                    images[:, :, imageNum] = depth_image

                self.c = np.asanyarray(ir1_frame.get_data())
                averageImage = np.median(images, axis=2)
                if not self.fromFile:
                    colorImage = np.asanyarray(color_frame.get_data())

                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                self.x , self.y , self.z = convert_depth_frame_to_pointcloud(averageImage , depth_intrin , depth_scale)

            finally:
                # Stop streaming
                pipeline.stop()
        time_end = datetime.now()
        print("capture time = ", time_end - time_start)



        # if (self.isFramos):
        #     colorImage = cv2.cvtColor(colorImage.astype(np.uint8) , cv2.COLOR_BGR2RGB)

        colorImage = cv2.cvtColor(colorImage.astype(np.uint8) , cv2.COLOR_BGR2RGB)
        plt.imshow(colorImage.astype(np.uint8))

        # self.mask2 = findMedia(self.c,colorImage, isBrown=self.isBrown, zImage=self.z)
        self.mask2 = np.ones(self.c.shape)

        if not self.fromFile:
            now = datetime.now()
            outfile = self.capt_saving_subfolder + '\\' + now.strftime("cloud_%d%m%Y_%H%M%S")
            np.savez(outfile, averageImage, self.c, colorImage, self.x, self.y, self.z)
        # fig = Figure(figsize=(5 , 4) , dpi=100)
        if self.fig is None:
            self.fig = plt.figure()
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
        else:
            self.clearPoly()

        overlay = np.copy(self.c)*self.mask2
        alpha = 0.8
        image_new = cv2.addWeighted(overlay , alpha , self.c , 1 - alpha , 0)
        self.ax.imshow(image_new)

        # self.ax.imshow(self.c)

        imageCanvas = self.builder.get_object('imageCanvas', self.mainwindow)
        imageCanvas.delete("all")

        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(self.fig , master=imageCanvas)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect('button_press_event' , self.on_key_press)
        # canvas.bind("<Button 1>" , self.on_key_press)

    def on_key_press(self, event):
        # print("you pressed {} {}".format(event.xdata,event.ydata))
        self.poly = np.vstack((self.poly , np.array([event.xdata , event.ydata] , dtype=np.int32)))

        if self.poly.shape[0]>2:
            overlay = np.copy(self.c)
            poly = np.int32([self.poly] , dtype=np.uint8)
            cv2.fillPoly(overlay , poly , color=0)
            alpha = 0.4
            image_new = cv2.addWeighted(overlay , alpha , self.c , 1 - alpha , 0)
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            self.ax.imshow(image_new)

        self.ax.scatter(self.poly[:,0] , self.poly[:,1] , s=10 , c='red' , marker='o')

        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def calibrate(self):
        """
        This describes the actions in an event of "calibration" button press.
        :param multi_image: int
            This sets whether to use or not (0 or else) a multi capturing calibration routine.
            A positive number sets the amount of captures.
        :return: void
            results are saved to the disk in a file
        """

        #     clean (tzvia)
        #  clean data -> calibration.cleanpointarray
        # call find plane coefficients
        # save_plane_info
        multi_image = self.cal_multi_image
        # TODO - Add a test that the "self.poly" is a valid collection of points
        # acquire the interest area from main screen "polyPoints"
        poly = np.int32([self.poly], dtype=np.uint8)
        cimg = np.zeros_like(self.z)
        cv2.fillPoly(cimg, poly, color=255)
        point_array = np.vstack(
            (self.x[cimg == 255].flatten(), self.y[cimg == 255].flatten(),
             self.z[cimg == 255].flatten())).transpose()

        if multi_image > 0:
            # init memory
            coef_arr = np.zeros([multi_image,3])
            # main capturing loop
            for i in range(multi_image):
                self.capture()
                point_array = np.vstack(
                    (self.x[cimg == 255].flatten(), self.y[cimg == 255].flatten(),
                     self.z[cimg == 255].flatten())).transpose()
                # Filtering switch
                if self.cal_use_filter_type == 'box':
                    filt = Filter()
                    filt.point_array = point_array
                    filt.box((-0.9, -0.9, 0.6), (1.8, 1.8, 1.0))
                    self.cal.clean_point_array = filt.point_array
                elif self.cal_use_filter_type == 'sphere':
                    raise NotImplementedError("Spherical filter is not implemented in caputeUI")
                    self.cal.clean_point_array = point_array
                elif self.cal_use_filter_type == 'None':
                    point_array = point_array[np.isfinite(point_array[: , 2]) , :]
                    self.cal.clean_point_array = point_array
                else:
                    csdata, C, X, Y, Z, r2, s = cs.fit_clean_surf(point_array, order=3, iter=5, coeff_mx_change=0.01)
                    self.cal.clean_point_array = csdata
                # Plane finder
                self.cal.find_plane_coefficients(use_ransac=self.cal_use_ransac)
                coef_arr[i, :] = np.asarray(self.cal.plane_coefficients)  # store last coeffs
            self.cal.plane_coefficients = tuple(np.mean(coef_arr, axis=0))

        else:
            # Filtering switch
            if self.cal_use_filter_type == 'box':
                filt = Filter()
                filt.point_array = point_array
                filt.box((-0.9, -0.9, 0.6), (1.8, 1.8, 1.0))
                self.cal.clean_point_array = filt.point_array
            elif self.cal_use_filter_type == 'sphere':
                raise NotImplementedError("Spherical filter is not implemented in caputeUI")
                self.cal.clean_point_array = point_array
            elif self.cal_use_filter_type == 'None':
                point_array = point_array[np.isfinite(point_array[: , 2]) , :]
                self.cal.clean_point_array = point_array
            else:
                csdata, C, X, Y, Z, r2, s = cs.fit_clean_surf(point_array, order=3, iter=5, coeff_mx_change=0.01)
                self.cal.clean_point_array = csdata
            self.cal.find_plane_coefficients(use_ransac=self.cal_use_ransac)
        self.cal.save_plane_info('generic_coef.json', method='json')

    def loadPoints(self):
        file_path = filedialog.askopenfilename()
        # TODO - add an if for empty path (cancel in "askopen") or none existing stuff
        f = np.load(file_path)
        print("file loaded is", file_path)
        self.poly = f['arr_0']
        coefficients = f['arr_1']
        self.cal.plane_coefficients = coefficients
        self.boxes = f['arr_2']
        self.redrawCanvas()

    def calcParams(self):
        # TODO - Take the next 3 text-lines and try to add the boxes to the calculated image
        # for rect in self.rects:
        #     self.ax.add_patch(rect)
        # self.ax.add_patch(self.rects[0])

        # shift measeuremenet boxes
        boxes = self.shiftBoxes()

        now = datetime.now()

        outfile = self.capt_saving_subfolder + '\\' + now.strftime("calc_%d%m%Y_%H%M%S")
        np.savez(outfile , self.poly , self.cal.plane_coefficients , boxes)

        poly = np.int32([self.poly] , dtype=np.uint8)
        cimg = np.zeros_like(self.z)
        if poly.size==0 and self.mask2 is not None:
            cimg[self.mask2]=255
        else:
            cv2.fillPoly(cimg , poly , color=255)



        # Access the image pixels and create a 1D numpy array then add to list
        # pts = np.where(cimg == 255)

        # systemCoeffs = np.asanyarray([0,0,1])

        # yangle = math.acos(
        #     np.dot(self.liftCoeffs[0::2] / np.sum(self.liftCoeffs[0::2] ** 2) ** 0.5 , systemCoeffs[0::2] / np.sum(systemCoeffs[0::2] ** 2) ** 0.5))
        # xangle = math.acos(
        #     np.dot(self.liftCoeffs[1:] / np.sum(self.liftCoeffs[1:] ** 2) ** 0.5 ,
        #            systemCoeffs[1:] / np.sum(systemCoeffs[1:] ** 2) ** 0.5))





        point_array = np.vstack((self.x[cimg==255].flatten(),self.y[cimg==255].flatten(),self.z[cimg==255].flatten())).transpose()
        point_array = point_array[point_array[:,2]>0,:]


        if self.cal_perform_rotation:  # for debug only!!!!!!  use False for real apps

            rotatedPoints = self.cal.rotate_about_calibration(point_array,
                                                              coefficients=None,
                                                              with_print_log=self.cal_display_angles,
                                                              dual_rotation=self.cal_perform_dual_rotation)
        else:
            rotatedPoints = point_array

        if False: # for debug only!!!!!!  use False for real apps
            filt = Filter()
            filt.point_array = rotatedPoints
            filt.box((-0.9, -0.9, 0.6),(1.8, 1.8, 1.0))
            tmp_var = filt.point_array
            # C, X, Y, Z, r2, s = cs.fit_surface(filt.point_array, order=3)
            polyPoints = np.array([[np.min(tmp_var[:, 0]), np.min(tmp_var[:, 1])],
                                   [np.max(tmp_var[:, 0]), np.max(tmp_var[:, 1])]])
            C, X, Y, Z, residuals = sfs.fit_surface(tmp_var, polyPoints, order=self.fit_order)

        else:
            # csdata, C, X, Y, Z, r2, s = cs.fit_clean_surf(rotatedPoints , order=3 , iter=5 , coeff_mx_change=0.01)

            csdata , C , res = sfnz.fit_clean_surf(rotatedPoints , order=3 , iter=5 , coeff_mx_change=0.01)

            if poly.size == 0 and self.mask2 is not None:
                polyPoints = np.array([[np.min(csdata[:,0]),np.min(csdata[:,1])],[np.max(csdata[:,0]),np.max(csdata[:,1])]])
            else:
                polyPoints = np.array([[np.min(rotatedPoints[:,0]),np.min(rotatedPoints[:,1])],[np.max(rotatedPoints[:,0]),np.max(rotatedPoints[:,1])]])

            from scipy.interpolate import LinearNDInterpolator

            cartcoord = list(zip(rotatedPoints[: , 0] , rotatedPoints[: , 1]))
            interp = LinearNDInterpolator(cartcoord , rotatedPoints[: , 2] , fill_value=0)

            X , Y = np.meshgrid(
                np.arange(np.min(rotatedPoints[::10 , 0]) + 0.001 , np.max(rotatedPoints[::10 , 0]) - 0.05 , 0.002) ,
                np.arange(np.min(rotatedPoints[::10 , 1]) + 0.001 , np.max(rotatedPoints[::10 , 1]) - 0.05 , 0.002))

            Z0 = interp(X , Y)

            p5 = np.percentile(Z0 , 5)
            p95 = np.percentile(Z0 , 95)

            Z0[Z0 < p5] = p5
            Z0[Z0 > p95] = p95

            outfile = self.capt_saving_subfolder + '\\' + now.strftime("interpolated_raw_%d%m%Y_%H%M%S.csv")
            np.savetxt(outfile , Z0 , delimiter=',')

            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111 , projection='3d')
            ax.scatter(rotatedPoints[::10 , 0] , rotatedPoints[::10 , 1] , rotatedPoints[::10 , 2])
            fig = plt.figure()
            ax = fig.add_subplot(111 , projection='3d')
            ax.scatter(csdata[::10 , 0] , csdata[::10 , 1] , csdata[::10 , 2])

            # C, X, Y, Z, residuals, A, axes = sfs.fit_surface(csdata , polyPoints , order=self.fit_order)
            # order=self.fit_order

            # C , X , Y , Z , order = bsfs.fit_auto_order(csdata , selected_point_list=polyPoints ,
            #                                            max_diff=self.order_max_dif)

            C , X , Y , Z , order = sfs.fit_auto_order(csdata , selected_point_list=polyPoints, max_diff=self.order_max_dif,max_diff_factor=self.max_diff_factor)

            # C , order = sfnz.fit_auto_order(csdata , max_diff=self.order_max_dif ,
            #                                            max_diff_factor=self.max_diff_factor)

            print("order selected: {}".format(order))

        # w , warp_type , frameMinIsGlobalMin , frameMaxIsGlobalMax = sfs.board_warp(Z, d1row=4, d2row=205, d1col=4, d2col=205)

        w , warp_type , frameMinIsGlobalMin , frameMaxIsGlobalMax = sfnz.board_warp(csdata, C , d1row=4 , d2row=205 , d1col=4 ,
                                                                                   d2col=205 , selected_point_list=polyPoints, res_X_Y=0.003)

        ztemp = np.max(Z) - Z

        plt.figure()
        plt.imshow(ztemp)
        # plt.show()

        outfile = self.capt_saving_subfolder + '\\' + now.strftime("surface_%d%m%Y_%H%M%S.csv")
        np.savetxt(outfile,ztemp,delimiter=',')

        # Z_mean_min_max = sfs.fit_Z_to_rectangles(Z, boxes)

        Z_mean_min_max = sfnz.fit_Z_to_rectangles(csdata, C, boxes, avg_high_perc = np.ones(len(boxes)), selected_point_list=polyPoints, board_height=None, X=None, Y=None, res_X_Y=0.001)

        Zlabel=''

        Z_mean_min_max = np.max(Z_mean_min_max)-Z_mean_min_max

        for index in range(len(Z_mean_min_max)):
            Zlabel = Zlabel+'Box:{}   Avg:{:3.0f}   Min:{:3.0f}   Max:{:3.0f}'.format(index+1,Z_mean_min_max[index][0]*1000,Z_mean_min_max[index][2]*1000,Z_mean_min_max[index][1]*1000)+'\n'

        Zlabel = Zlabel+'warp: {:1.3f}'.format(w*1000)

        self.builder.get_object('rectangles_label' , self.mainwindow)['text'] = Zlabel
        # Addition added to test with alex the accuracy of the method - operate from "config.ini" !!!
        if poly.size > 0:
            if self.save_line:
                line_ex = LineExtractor()
                line_ex.point_cloud_calc = [X, Y, Z]
                line_ex.point_cloud_meas = [self.x, self.y, self.z]
                line_ex.pt = poly[0, 0]  # First pair of points
                line_ex.pt_meter_meas = (self.x[line_ex.pt[1], line_ex.pt[0]],
                                         self.y[line_ex.pt[1], line_ex.pt[0]],
                                         self.z[line_ex.pt[1], line_ex.pt[0]])  # find value in [meters]
                tmp = np.asarray(line_ex.pt_meter_meas).reshape([1,3])
                tmp = self.cal.rotate_about_calibration(tmp, dual_rotation=self.cal_perform_dual_rotation)
                line_ex.pt_meter_calc = tuple(tmp[0])  # value in [meters] rotated
                try:
                    line_ex.get_y_line(data_source='calc')
                    line_ex.get_x_line(data_source='calc')
                    line_ex.get_y_line(data_source='meas')
                    line_ex.get_x_line(data_source='meas')
                except ():
                    raise FileNotFoundError("Problem with generating the line-file")

    def redrawCanvas(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)

        self.ax.imshow(self.c)
        self.ax.scatter(self.poly[: , 0] , self.poly[: , 1] , s=10 , c='red' , marker='o')

        imageCanvas = self.builder.get_object('imageCanvas', self.mainwindow)
        imageCanvas.delete("all")

        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(self.fig, master=imageCanvas)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect('button_press_event', self.on_key_press)

    def shiftBoxes(self):
        beltEntryVar = self.builder.get_variable('beltEntryVar')#  , self.mainwindow)
        shift = beltEntryVar.get()
        boxes = np.copy(self.boxes)

        for boxNum in self.boxes_to_shift:
            boxes[boxNum-1][2] += shift
            boxes[boxNum - 1][3] += shift

        return boxes

def on_closing():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        if app.capt_use_live_capture and (app.capt.pipe is not None):
            app.capt.stop()
        root.destroy()
        root.quit()

        return


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    exit()
