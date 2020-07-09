import json
import cv2
from pathlib import Path
import numpy as np
import time

# from lib.point_filters import Filter as lf
# from lib import helpers as lh


class Calibration:
    #property

    #classmethod

    def __init__(self):
        self.full_point_cloud = None  # Not used under the captureUI
        self.clean_point_cloud = None  # point cloud after initial reduction - ELAD, check if that even possible # Not used under the captureUI
        self.clean_point_array = None  # point array after reduction and filtering
        self.plane_coefficients = (0, 0, 0)  # (C(x), C(y), C(1))
        self.origin_location = (0, 0)  # (X, Y)
        self.contour_list_ind = []  # in indices units

        # Not used under the captureUI
        self.angle_about_X = 0.0  # in degrees
        self.angle_about_Y = 0.0  # in degrees
        self.angle_about_Z = 0.0  # in degrees

        self.angle_for_major = 0.0  # rotation about X, in RAD
        self.angle_for_minor = 0.0  # rotation about Y, in RAD

    def choose_points_V1(self, values, method='rectangle'):  # PROBLEM
        # this method has a problem because the units on the points in the point array are in meters and not in indexs
        # as the "cv2.pointPolygonTest" woud require (due to the use of contours in indices units)
        """
        takes the points that are suppose to be in the shape of interest.
        THIS ALL HAPPENS IN THE XY PLANE
        """
        # TODO - Add a checkup weather the point array even have points
        if len(self.contour_list_ind):
            raise NotImplementedError("It is not possible to add contours at the moment")
        if method == 'rectangle':
            x = values[0][0]
            y = values[0][1]
            w = values[1]
            h = values[2]
            # [(x,y),w,h] from top left -> counter-clockwise
            self.contour_list_ind.append((x, y))
            self.contour_list_ind.append((x, y + h))
            self.contour_list_ind.append((x + w, y + h))
            self.contour_list_ind.append((x + w, y))
        elif method == 'list':
            # TODO - test if this list is indeed a valid list
            self.contour_list_ind.extend(values)
        else:
            raise NotImplementedError("Method of segmenting a contour is invalid")
        new_array = np.zeros(self.clean_point_array.shape)
        cnt = 0
        for pt in self.clean_point_array:
            if cv2.pointPolygonTest(self.contour_list_ind, (pt[0], pt[1]), measureDist=False):
                new_array[cnt] = pt
                cnt += 1
        self.clean_point_array = new_array[0:cnt]

    def choose_points_V2(self, values, method='rectangle'):  # This one uses indices units as appose to V1
        """
        takes the points that are suppose to be in the shape of interest.
        THIS ALL HAPPENS IN THE XY PLANE
        """
        # TODO - Add a checkup weather the point-cloud even have points
        if len(self.contour_list_ind):
            raise NotImplementedError("It is not possible to add contours at the moment")
        if method == 'rectangle':
            x = values[0][0]
            y = values[0][1]
            w = values[1]
            h = values[2]
            # [(x,y),w,h] from top left -> counter-clockwise
            self.contour_list_ind.append([x, y])
            self.contour_list_ind.append([x, y + h])
            self.contour_list_ind.append([x + w, y + h])
            self.contour_list_ind.append([x + w, y])
        elif method == 'list':
            # TODO - test if this list is indeed a valid list
            self.contour_list_ind.extend(values)
        else:
            raise NotImplementedError("Method of segmenting a contour is invalid")

        contour_mat = np.asarray(self.contour_list_ind)
        new_array = np.zeros([len(self.full_point_cloud[0].flatten()), 3])
        [lim_y, lim_x] = self.full_point_cloud[0].shape
        cnt = 0
        ind_x = 0
        ind_y = 0
        while ind_x < lim_x:
            while ind_y < lim_y:
                if cv2.pointPolygonTest(contour_mat, (ind_y, ind_x), measureDist=False): # This returns true all the time!!!!!
                    new_array[cnt] = np.array([self.full_point_cloud[0][ind_y, ind_x],
                                               self.full_point_cloud[1][ind_y, ind_x],
                                               self.full_point_cloud[2][ind_y, ind_x]])
                    cnt += 1
                ind_y += 1
            ind_x += 1

        self.clean_point_array = new_array[0:cnt]

    #!!!!!!!!!!!!! DEPRACATED !!!!!!!!!!!!!!!!!!!!
    # def clean_pointcloud_store_pointarray(self, patch, expected_dist=(0.2, 4.5)):
    #     """
    #     method implements an initial cleanup of the point-cloud and storage in a point-array
    #     Params:
    #        patch : ((x, y), w, h)
    #            The patch as taken by the image.
    #        expected_dist : Tuple(double, double)
    #            The two spherical distances of the expected points.
    #     Returns:
    #        void
    #     """
    #     # filter = lf()
    #     # lf.is_valid_data()
    #     # TODO - transfer to the "Filter" class
    #     self.clean_point_array = lh.filter_point_cloud(self.full_point_cloud, patch, expected_dist)
    #     raise NotImplementedError

    def find_plane_coefficients(self, use_ransac=False):
        """
        Retrieve parameters of plane equation from a point-array
            details are in the class-method "find_surf_point_array"

        Returns:
            void
        """

        # Test for data integrity is in the class-method "find_surf_point_array"
        if use_ransac:
            xcrop = self.clean_point_array[:, 0]
            ycrop = self.clean_point_array[:, 1]
            zcrop = self.clean_point_array[:, 2]
            # X = np.vstack((xcrop.flatten(), ycrop.flatten()))
            X = np.vstack((xcrop, ycrop))

            from sklearn import linear_model

            # Init
            ransac = linear_model.RANSACRegressor()
            ransac.min_samples = 0.1 # 20% of points inserted

            # ransac.fit(X.transpose(), zcrop.flatten().transpose())
            ransac.fit(X.transpose(), zcrop.transpose())

            ransacPred = ransac.predict(X.transpose())
            # resid = zcrop - ransacPred.reshape(zcrop.shape) # Residue testing
            self.clean_point_array = np.vstack((xcrop, ycrop, ransacPred)).transpose()

        self.plane_coefficients = self.find_surf_from_point_array(self.clean_point_array)#(self.clean_point_array)#self.find_surf_point_array(self.clean_point_array)

        return

    def find_origin(self, image):
        """
        method does ...
        Params:
            param1 : type
                Explenation.
            param2 : type
                Explenation.
        Returns:
            void
        """
        if len(image.shape) == 3:
            # treat as RGB
            raise NotImplementedError
        else:
            # treat as IR
            raise NotImplementedError

    def save_plane_info(self, fullname, method='json'):
        """
        Saves the data of the calibration frame: 3 coefficients of the calibration plane and
        location of origin in space in the camera frame
        Params:
            fullname : string
                The full path and file name to be loaded.
        Returns:
            bool, T/F according to the success of the save
        """
        fullname = Path(fullname)
        fullname.with_suffix('.' + method)
        info = {'coefficients': tuple(self.plane_coefficients),
                'origin': tuple(self.origin_location)}
        if method == 'json':
            # fullname += '.json'
            # Serialize data into file:
            try:
                with open(fullname, mode='w') as file:
                    json.dump(info, file)
                print("File was saved as:\n{0}\n".format(fullname))
                return True
            except():
                raise RuntimeError("Something didn't work with the save!")
                return False
        elif method == 'ini':
            raise NotImplementedError("Saving an ini file is not available yet.")
            return False
        else:
            raise RuntimeError("The saving method requested is not valid.")
            return False

    def load_plane_info(self, fullname, method='json'):
        """
        Loads the data of the calibration frame: 3 coefficients of the calibration plane and
        location of origin in space in the camera frame
        Params:
            fullname : string
                The full path and file name to be loaded.
        Returns:
            bool, T/F according to the success of the load
        """
        # TODO - Check if it has '.json' at the end
        fullname = Path(fullname)
        fullname.with_suffix('.' + method)
        if not fullname.exists():
            raise RuntimeError("The provided \"fullname\" doesn't exist!")
            return False

        if method == 'json':
            try:
                with open(fullname, mode='r') as file:
                    info = json.load(file)
                self.plane_coefficients = info['coefficients']
                self.origin_location = info['origin']
                print("File was loaded from:\n{0}\n".format(fullname))
                return True
            except():
                print("Something didn't work with the load!")
                return False
        elif method == 'ini':
            raise NotImplementedError("Loading an ini file is not available yet.")
            return False
        else:
            raise RuntimeError("The loading method requested is not valid.")
            return False

    def rotate_about_calibration(self, points, coefficients=None, with_print_log=False, dual_rotation=False, calc_method='base'):
        """
        Method rotates a point-array into the calibration plane system.
        It takes the coefficients for the plane of reference [x0, y0, D] (where: Z = x0*X + y0*Y + D)

        Params:
            points : point_array
                The points of the object aimed to transfer into the machine system.
            coefficients: None or Tuple(double, double, double)
                This is an override for the class defined coefficients
            with_print_log : bool
                This is a sort of debugging mechanism.
                Prints out the 3 angles of the normal with principal axises.
            dual_rotation : bool
                Sets an extra rotation about the Y Axis.

        Returns:
            new_points: point_array
        """
        # Init
        from scipy.spatial.transform import Rotation as Rot
        # TODO - attend to what you get - check validity of  point-array
        if coefficients is None:
            coefficients = tuple(self.plane_coefficients)
        if coefficients == (0, 0, 0):
            raise RuntimeError("Plane coefficients are not valid!!!!")

        # TODO - Implement the translation action
        feeder_translation = [0, 0, 0]  # 24.02.2020, Elad: Not yet implemented

        plane_normal = self.coeff2normal(coefficients)

        if not (feeder_translation == [0, 0, 0]):
            raise NotImplementedError("'feeder_translation' option is not available yet!")

        withX = np.arccos(np.dot(plane_normal, [1, 0, 0]))  # in RAD
        withY = np.arccos(np.dot(plane_normal, [0, 1, 0]))  # in RAD
        withZ = np.arccos(np.dot(plane_normal, [0, 0, 1]))  # in RAD
        # tmpVec = plane_normal*[1, 0, 1]/np.sqrt(plane_normal[0]**2+plane_normal[2]**2)
        tmpVec = np.array([plane_normal[0], 0, plane_normal[2]]) / np.sqrt(pow(plane_normal[0], 2) + pow(plane_normal[2], 2))
        withXZ = np.arccos(np.dot(plane_normal, tmpVec))  # in RAD
        if with_print_log:
            print("Angle with X axis", withX * 180 / np.pi, "°")
            print("Angle with Y axis", withY * 180 / np.pi, "°")
            print("Angle with Z axis", withZ * 180 / np.pi, "°")
            print("Angle with XZ plane", withXZ * 180 / np.pi, "°")

        # This preforms a very basic rotation around one axis at a time!!!!
        # TODO - Add more rotations or create one uniform rotation - ELAD 24.02.2020

        # 12.03.2020 - Rotation was changed to "withXZ"

        if plane_normal[1] < 0:  # Tells whether to turn clockwise or counter_clockwise
            self.angle_for_major = -withXZ  # in our system negative Y is negative rotation angle about X
        else:
            self.angle_for_major = withXZ  # in our system positive Y is positive rotation angle about X
        r_x = Rot.from_rotvec([self.angle_for_major, 0, 0])
        if dual_rotation:
            tmp_normal = r_x.apply(plane_normal)  # finds the new angle
            self.angle_for_minor = np.arccos(np.dot(tmp_normal, [0, 0, 1]))
            if tmp_normal[0] > 0:  # Tells whether to turn clockwise or counter_clockwise
                self.angle_for_minor = -self.angle_for_minor
        else:
            self.angle_for_minor = 0.0
        r_y = Rot.from_rotvec([0, self.angle_for_minor, 0])

        # actual rotation
        t = time.time()

        new_points = np.zeros(points.shape)
        if calc_method == 'base':

            new_points = r_x.apply(points)
            new_points = r_y.apply(new_points)

            # new_points2 = np.zeros(points.shape)
            # tmp_point = np.zeros([1 , 3])
            # for pt_ind in range(points.shape[0]):
            #     tmp_point = r_x.apply(points[pt_ind , :])
            #     new_points2[pt_ind , :] = r_y.apply(tmp_point)
            #
            # print ("max: " + str(np.max(new_points-new_points2)))

        elif calc_method == 'single_call':
            for pt_ind in range(points.shape[0]):
                new_points[pt_ind, :] = r_y.apply(r_x.apply(points[pt_ind, :]))
        elif calc_method == 'parafor':
            raise NotImplementedError(f"Calculation method \"{calc_method}\" is not implemented!")
            # for pt_ind in range(points.shape[0]):
            #     new_points[pt_ind, :] = r_y.apply(r_x.apply(points[pt_ind, :]))
        elif calc_method == 'single_mat':
            raise NotImplementedError(f"Calculation method \"{calc_method}\" is not implemented!")
            new_points = None
        else:
            raise RuntimeError(f"Calculation method \"{calc_method}\" is not valid!")
            new_points = None


        # print ("elapsed: "+ str(time.time() - t))

        return new_points



    # !!!!!!!!!!!! DEPRECATED !!!!!!!!!!!!!!!!!
    # def rotate_calibration(self, points, coefficients=None, with_print_log=False, dual_rotation=False,
    #                        rotate_about=(1, 0, 0)):
    #     """
    #     Method rotates a point-array into the calibration plane system.
    #     It takes the coefficients for the plane of reference [x0, y0, D] (where: Z = x0*X + y0*Y + D)
    #
    #     Params:
    #         points : point_array
    #             The points of the object aimed to transfer into the machine system.
    #         coefficients: None or Tuple(double, double, double)
    #             this is an override for the class defined coefficients
    #         with_print_log : bool
    #             This is a sort of debugging mechanism.
    #             Prints out the 3 angles of the normal with principal axises.
    #         dual_rotation : bool
    #             Sets an extra rotation about the Y Axis.
    #         rotate_about : Tuple(bool, bool, bool)
    #             Sets the axis of convergence about.
    #
    #     Returns:
    #         new_points: point_array
    #     """
    #     # Init
    #
    #     from scipy.spatial.transform import Rotation as Rot
    #     # TODO - attend to what you get - point-cloud or point-array
    #     if not coefficients:
    #         coefficients = self.plane_coefficients
    #         feeder_coeffs = self.plane_coefficients
    #     if coefficients == (0, 0, 0):
    #         raise RuntimeError("Plane coefficients are not valid!!!!")
    #
    #     stack_points = points
    #     # TODO - Implemnt the movement action
    #     feeder_translation = [0, 0, 0]  # 24.02.2020, Elad: Not yet implemented
    #
    #     coeff_normal_lift = self.coeff2normal(feeder_coeffs)
    #
    #     if not (feeder_translation == [0, 0, 0]):
    #         raise NotImplementedError("'feeder_translation' option is not available yet!")
    #
    #     withX = np.arccos(np.dot(coeff_normal_lift, [1, 0, 0]))  # *180/np.pi
    #     withY = np.arccos(np.dot(coeff_normal_lift, [0, 1, 0]))  # *180/np.pi
    #     withZ = np.arccos(np.dot(coeff_normal_lift, [0, 0, 1]))  # *180/np.pi
    #     if with_print_log:
    #         print("Angle with X axis", withX * 180 / np.pi, "°")
    #         print("Angle with Y axis", withY * 180 / np.pi, "°")
    #         print("Angle with Z axis", withZ * 180 / np.pi, "°")
    #
    #     # This preforms a very basic rotation around one axis at a time!!!!
    #     # TODO - Add more rotations or create one uniform rotation - ELAD 24.02.2020
    #
    #     if dual_rotation:
    #         r = Rot.from_rotvec([withZ, 0, 0])
    #     else:
    #         # TODO - insert test for the "rotvec"
    #         rotvec = np.asarray(rotate_about, type=bool) * np.asarray([withZ, withZ, withZ])
    #         r = Rot.from_rotvec(rotvec)
    #     new_stack_points = np.zeros(stack_points.shape)
    #     for pt_ind in range(stack_points.shape[0]):
    #         new_stack_points[pt_ind, :] = r.apply(stack_points[pt_ind, :])
    #
    #     if dual_rotation:
    #         dual_rotation_coeffs = self.find_surf_point_array(new_stack_points)
    #         new_stack_points = self.rotate_about_calibration(new_stack_points,
    #                                                          coefficients=dual_rotation_coeffs,
    #                                                          with_print_log=with_print_log,
    #                                                          dual_rotation=False,
    #                                                          rotate_about=(0, 1, 0))
    #     new_points = new_stack_points
    #
    #     return new_points

    @staticmethod
    def coeff2normal(coefficient_1st_order):
        """
        Generate a normal vector to a plane equation coefficients
        Under the assumption that:
        Z = c0*X + c1*Y + c2
       Params:
            coefficient_1st_order: Tuple(double, double, double)
                coefficient_1st_order = (c0, c1, c2)
        Returns:
            normal: Tuple(double, double, double)
                The normal to the plane where (x, y, z)
        """
        # TODO - make test functions to make sure the input is correct
        c = coefficient_1st_order
        normal = (-c[0] / np.sqrt(c[0] ** 2 + c[1] ** 2 + 1),
                  -c[1] / np.sqrt(c[0] ** 2 + c[1] ** 2 + 1),
                  1 / np.sqrt(c[0] ** 2 + c[1] ** 2 + 1))
        return normal

    @staticmethod
    def find_surf_from_point_array(point_array):
        """
        Retrieve parameters of plane equation from a point-array
            After getting the Coefficients you can:
            evaluate it on grid:
               Z = C[0]*X + C[1]*Y + C[2]
            where X, Y are mesh-grids
            or expressed using matrix/vector product
                XX = X.flatten()
                YY = Y.flatten()
                Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

            # point_array shape:    x0,y0,z0
            #                       x1,y1,z1
            #                         ...
            #                       xn,yn,zn

        Returns:
            coefficients: Tuple(C[0], C[1], C[2])
        """
        # Init tests
        if point_array is None:
            raise RuntimeError("\"self.clean_point_array\" is empty")
            return
        elif point_array.shape[0] <= 3 or point_array.shape[1] != 3:
            raise RuntimeError("\"self.clean_point_array\" is not the correct shape/size")
            return

        input_matrix = np.c_[point_array[:, 0],
                             point_array[:, 1],
                             np.ones(point_array.shape[0])]  # concatenation along the second axis
        coefficients, _, _, _ = np.linalg.lstsq(input_matrix, point_array[:, 2])

        return tuple(coefficients)



