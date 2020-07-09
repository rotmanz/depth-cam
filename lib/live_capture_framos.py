import os
from time import time
from datetime import datetime
import numpy as np
import pyrealsense2 as rs

import lib.helpers as lh


class LiveCapture:
    # consts
    align_to_depth = rs.stream.depth  # pointer to "Whom to allign to"
    align_to_color = rs.stream.color  # pointer to "Whom to allign to"

    def __init__(self, align=0, path="", filename=""):
        """

        :param align: int
            Switch that chooses what image to allign to:
            0 - depth
            1 - color
        :param path:
        :param filename:
        """
        if align == 0:
            self.align = rs.align(LiveCapture.align_to_depth)
        elif align == 1:
            self.align = rs.align(LiveCapture.align_to_color)
        else:
            raise RuntimeWarning("Chosen alignment is illegal - alignment set to depth")
            self.align = rs.align(LiveCapture.align_to_depth)
        # TODO - Take next line out to config.ini file
        self.IR_cam = 0  # camera index (0 or 1)
        self.filename = filename
        if path:
            self.path = path
        else:
            self.path = os.getcwd()
        # TODO - Take next line out to config.ini file
        self.im_format = 'TIFF'  # see https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for more options
        self.device_profile = None
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        # Saved files to return to
        # TODO - Take next line out to config.ini file
        self.np_filename = ""
        self.color_filename = ""
        self.ir_filename = ""
        self.resX  = 640
        self.resY = 480
        self.frameRate=30

    def start(self):
        """
        Initiate the pipeline
        :return: void
        """
        # self.cfg.enable_record_to_file(self.filename)
        # self.cfg.enable_all_streams()
        self.cfg.enable_stream(rs.stream.depth , self.resX , self.resY , rs.format.z16 , self.frameRate)
        self.cfg.enable_stream(rs.stream.color , self.resX , self.resY , rs.format.bgr8 , self.frameRate)
        self.cfg.enable_stream(rs.stream.infrared , 1 , self.resX , self.resY , rs.format.y8 , self.frameRate)
        self.cfg.enable_stream(rs.stream.infrared , 2 , self.resX , self.resY , rs.format.y8 , self.frameRate)
        # self.cfg.enable_device()
        if self.cfg.can_resolve(self.pipe):
            self.cfg.resolve(self.pipe)
            self.device_profile = self.pipe.start(self.cfg)
            print("Pipeline has started")
            if False: # TODO - This is a placeholder for setting te laser power - need tests before integration
                device = self.device_profile.get_device()
                depth_sensor = device.query_sensors()[0]
                laser_range = depth_sensor.get_option_range(rs.option.laser_power)
                # TODO - Take next line out to config.ini file
                depth_sensor.set_option(rs.option.laser_power, laser_range.max)
        else:
            print("Device can not be resolved")
            raise ConnectionError("Camera not present or something is wrong with the streams definition")
        return

    def stop(self):
        """
        Stop the pipeline streaming.
        :return: void
        """
        self.pipe.stop()
        print("Pipeline has stopped")

    def capture_one(self, no_attempts: int = 5, timeout_ms : int = 1000):
        """
        As name suggests - function captures one image from all sensors

        :param no_attempts: int
            Number of attempts to capture the image.
        :return depth_frame, ir_frame, color_frame:
            3 frame that are numpy array ([h,w], [h,w], [h,w,3])
        """
        # frames = self.pipe.wait_for_frames()
        success = False
        cnt = 0
        while (not success) and cnt < no_attempts:
            # TODO - Take next line out to config.ini file - "timeout_ms"
            (success, frames) = self.pipe.try_wait_for_frames(timeout_ms=timeout_ms)
            cnt += 1
        if success:
            aligned_frames = self.align.process(frames)
        else:
            raise RuntimeError("problem with capturing frames")
        try:
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())
        except():
            depth_frame = np.zeros([480, 640])
            print("something went wrong with depth frame")

        try:
            ir_frame = aligned_frames.get_infrared_frame(self.IR_cam)
            ir_frame = np.asanyarray(ir_frame.get_data())
        except():
            ir_frame = np.zeros([480, 640])
            print("something went wrong with IR frame")

        try:
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
        except():
            color_frame = np.zeros([480, 640, 3])
            print("something went wrong with color frame")

        return depth_frame, ir_frame, color_frame

    def capture_average_reading(self, dry_runs=5, images_2_mean_count=10, mean_flag=False, single_capture_attempts=5, single_capture_timeout_ms=1000):
        """
        :param dry_runs: int
            amount of feeds taken from the camera as warm-up
        :param images_2_mean_count: int
            amount of images to combine as one for later processing
        :param mean_flag: bool
            States the kind of "mean".
                True = mean
                False = median
        :return: dict
            all_data is a dict with the fallowing keys:
            'X', 'Y','Z'  - 3 point-cloud components
            'ir': average_ir_image - aligned
            'color': average_color_image - aligned
        """
        all_data = {}
        # Fetch stream profile for depth stream
        depth_stream = self.device_profile.get_stream(rs.stream.depth)
        # Downcast to video_stream_profile and fetch intrinsics
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        # Extract depth scale
        depth_scale = self.device_profile.get_device().first_depth_sensor().get_depth_scale()

        for i in range(dry_runs):
            try:
                tmp_f_d, _, _ = self.capture_one()
            except RuntimeError as e:
                continue

        [res_y, res_x] = tmp_f_d.shape
        depth_array = np.zeros((res_y, res_x, images_2_mean_count))
        ir_array = np.zeros((res_y, res_x, images_2_mean_count))
        color_array = np.zeros((res_y, res_x, 3, images_2_mean_count),dtype=np.uint8)

        for i in range(images_2_mean_count):
            depth_array[:, :, i], ir_array[:, :, i], color_array[:, :, :, i] = self.capture_one(no_attempts=single_capture_attempts, timeout_ms=single_capture_timeout_ms)

        if mean_flag:
            average_depth_image = np.mean(depth_array, axis=2)
            average_ir_image = np.mean(ir_array, axis=2)
            average_color_image = np.mean(color_array, axis=3)
        else:
            average_depth_image = np.median(depth_array, axis=2)
            average_ir_image = np.median(ir_array, axis=2)
            average_color_image = np.median(color_array, axis=3)

        x, y, z = lh.convert_depth_frame_to_pointcloud(average_depth_image, depth_intrinsics, depth_scale)

        all_data = {'X': x,
                    'Y': y,
                    'Z': z,
                    'ir': average_ir_image,
                    'color': average_color_image,
                    'depth_raw': average_depth_image}
        return all_data

    def save_2_disk(self, data_dict, filename="", path=""):
        from PIL import Image

        dt = datetime.now()
        now_string = "{:02}{:02}{:04}_{:02}{:02}{:02}_".format(dt.day, dt.month, dt.year, dt.hour, dt.minute, dt.second)
        if path:
            old_path = self.path
            self.path = path
        if filename:
            old_filename = self.filename
            self.filename = (now_string + filename)
        else:
            old_filename = self.filename
            self.filename = (now_string + self.filename)
        # save the pointcloud
        pointcloud = [data_dict['X'], data_dict['Y'], data_dict['Z']]
        fullpath = self.path + '\\' + self.filename
        if lh.save_pointcloud(pointcloud, filename=fullpath):
            self.np_filename = fullpath + '.npz'
            # print is included in the save func already

        # save the images
        fullpath = self.path + '\\' + self.filename + '_ir.' + self.im_format
        im = Image.fromarray(data_dict['ir'].astype(np.uint8))#, mode="L")
        try:
            im.save(fullpath, format=self.im_format)
            self.ir_filename = fullpath
            print("IR image was saved")
        except():
            print("IR image could not be saved!!")
        fullpath = self.path + '\\' + self.filename + '_color.' + self.im_format
        im = Image.fromarray(data_dict['color'].astype(np.uint8))#, mode="RGB")
        try:
            im.save(fullpath, format=self.im_format)
            self.color_filename = fullpath
            print("color image was saved")
        except():
            print("color image could not be saved!!")

        # return to what it was
        self.filename = old_filename
        if path:
            self.path = old_path

        return
