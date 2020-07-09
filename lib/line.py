from pathlib import Path
from datetime import datetime

import numpy as np


class LineExtractor:
    def __init__(self):
        self.point_cloud_meas = None
        self.point_cloud_calc = None

        now = datetime.now()
        outfile = now.strftime("line_%d%m%Y_%H%M%S")
        self.fullname = (Path.cwd() / 'lines' / outfile).resolve()  # Keep it simple because later more details are added

        self.point_1 = (0.0, 0.0)  # upper point
        self.point_2 = (0.0, 0.0)  # lower point

        self.pt = (0, 0)  # first pair of points in pixels
        self.x = 0.0
        self.y = 0.0
        self.pt_meter_meas = (0.0, 0.0, 0.0)  # (x,y,z) in Meters
        self.pt_meter_calc = (0.0, 0.0, 0.0)  # (x,y,z) in Meters

        self.gap = 0.005  # in mm

    def get_y_line(self, data_source='meas'):
        """


        """
        # Inits

        if data_source == 'meas':
            pt_cl = self.point_cloud_meas  # pt_cl == point-cloud
            self.x = self.pt_meter_meas[0]
        elif data_source == 'calc':
            pt_cl = self.point_cloud_calc  # pt_cl == point-cloud
            self.x = self.pt_meter_calc[0]
        else:
            raise RuntimeError(f'The data source \"{data_source}\" is not valid!')
            return

        # self.x = self.point_cloud_meas[0][self.pt[1], self.pt[0]]  # will have problems with large tilt
        # l_boarder = self.x - self.gap
        # r_boarder = self.x + self.gap
        l_boarder = self.x - self.gap
        r_boarder = self.x + self.gap
        # Not sure if needed
        if l_boarder > r_boarder:
            tmp = r_boarder
            r_boarder = l_boarder
            l_boarder = tmp

        filename = str(self.fullname) + '_y-dir_' + data_source + '.csv'
        # extract points with similar X value
        res = np.where(np.logical_and((l_boarder < pt_cl[0]), (pt_cl[0] < r_boarder)))
        ans = np.c_[np.array(pt_cl[0][res]),
                    np.array(pt_cl[1][res]),
                    np.array(pt_cl[2][res])]
        np.savetxt(filename, ans, fmt='%.18e', delimiter=',')
        return

    def get_x_line(self, data_source='meas'):
        """


        """
        # Inits
        if data_source == 'meas':
            pt_cl = self.point_cloud_meas  # pt_cl == point-cloud
            self.y = self.pt_meter_meas[1]
        elif data_source == 'calc':
            pt_cl = self.point_cloud_calc  # pt_cl == point-cloud
            self.y = self.pt_meter_calc[1]
        else:
            raise RuntimeError(f'The data source \"{data_source}\" is not valid!')
            return

        # self.x = self.point_cloud_meas[0][self.pt[1], self.pt[0]]  # will have problems with large tilt

        l_boarder = self.y - self.gap
        r_boarder = self.y + self.gap
        # Not sure if needed
        if l_boarder > r_boarder:
            tmp = r_boarder
            r_boarder = l_boarder
            l_boarder = tmp

        filename = str(self.fullname) + '_x-dir_' + data_source + '.csv'
        # extract points with similar Y value
        res = np.where(np.logical_and((l_boarder < pt_cl[1]), (pt_cl[1] < r_boarder)))
        ans = np.c_[np.array(pt_cl[0][res]),
                    np.array(pt_cl[1][res]),
                    np.array(pt_cl[2][res])]
        np.savetxt(filename, ans, fmt='%.18e', delimiter=',')
        return
