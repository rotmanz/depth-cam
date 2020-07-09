import numpy as np


class Filter:
    """
    a class that is designed to filter point-arrays
    """
    def __init__(self):
        self.point_array = None

    def is_valid_data(self):
        """
        test the validity of the point-array
        """
        if self.point_array is not None:
            if type(self.point_array) is np.ndarray:
                if self.point_array.shape[1] == 3:
                    return True
        return False

    def spherical(self, expected_radii=(0.2, 4.5)):
        """
        filters out anything not between to two radii in 'expected_dist'.

        input:
        self.point_array - the point-array to filter
        expected_dist - (r0, r1) in [m]. The order of them doesn't matter

        output:
        self.point_array
        """
        if not self.is_valid_data():
            raise ValueError("Data in 'point_array' doesn't match criteria.")
        if (not type(expected_radii) is tuple) and (len(expected_radii) == 2) and (expected_radii >= (0, 0)):
            raise ValueError("Data in 'expected_dist' doesn't match criteria.\n"
                             "Input should be tuple of size 2, both >= 0.")
        # min-max test
        if expected_radii[1] < expected_radii[0]:
            expected_radii = (expected_radii[1], expected_radii[0])

        point_array_filtered = np.zeros(self.point_array.shape)  # Preparation
        cnt = 0
        for pt_ind in range(self.point_array.shape[0]):
            norm_cam_dist = np.linalg.norm(self.point_array[pt_ind, :])  # normalized distance from cam. origin
            if expected_radii[0] <= norm_cam_dist <= expected_radii[1]:
                point_array_filtered[cnt, :] = self.point_array[pt_ind, :]
                cnt += 1
        self.point_array = point_array_filtered[0:cnt - 1, :]  # remove additional zero cells
        return

    def box(self, top_left_close, boundaries):
        """
        filters out anything outside of the box
        input:
        self.point_array - the point-array to filter
        top_left_close - (x, y, z) the point of the start of the box (if needed, negative numbers go here)
        boundaries - (w, h, d)

        output: void
            saves the data to self.point_array
        """
        if not self.is_valid_data():
            raise ValueError("Data in 'point_array' doesn't match criteria.")
        if not type(top_left_close) is tuple and len(top_left_close) == 3:
            raise ValueError("Data in 'top_left_close' doesn't match criteria.\nInput should be tuple of size 3.")
        if not type(boundaries) is tuple and len(boundaries) == 3:
            raise ValueError("Data in 'boundaries' doesn't match criteria.\nInput should be tuple of size 3.")

        point_array_filtered = np.zeros(self.point_array.shape)  # Preparation
        cnt = 0
        for pt_ind in range(self.point_array.shape[0]):
            in_square_flag = True
            for i in range(3):
                if top_left_close[i] <= self.point_array[pt_ind, i] <= (top_left_close[i] + boundaries[i]):
                    continue
                else:
                    in_square_flag = False
                    break
            if in_square_flag:
                point_array_filtered[cnt, :] = self.point_array[pt_ind, :]
                cnt += 1
        self.point_array = point_array_filtered[0:cnt - 1, :]
        return
