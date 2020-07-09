import numpy as np
import scipy as sp
from scipy.spatial import distance_matrix
# import sympy as sym



def fit_surface(data, order=1, A=None):
    """
    input data: clean point cloud
    do 1st order fit (will support orders 1-7)
    return C, X, Y, Z, residuals
    return C: coefficient list. length is order dependent.
    order --> C.shape[0]: 0.5 * (2 + order)! / order!
    1 --> 3
    2 --> 6
    3 --> 10
    4 --> 15
    5 --> 21
    6 --> 28
    7 --> 36

    X,Y,Z mesh grid. Xmin-max, Ymin-max taken from list of user selected points (or from data, if there are no selected
    points).
    If board height is given as input, X range will be equal to input height and centered around average of
    selected points/data min/max values
    X and Y mesh with 1mm increments.
    Z is calculated value for given X,Y

    :param data: input clean point cloud [N,3]
    :param order: of surface fit. Default 3rd order
    :param selected_point_list: input np array [N,2] of user selected points
    :return: C: coefficient list. order dependent. For 3rd order fit: [C0, Cx1, Cy1, Cxy, Cx2, Cy2, Cx2y1, Cx1y2, Cx3, Cy3]
    X,Y,Z. Dimensions of X,Y,Z: [# incr Y, # incr X]
    """

    if A is None:
        baseunit = np.c_[data[:, :2]]
        order2_x1y1 = np.prod(np.c_[baseunit], axis=1)
        order3_x2y1 = np.prod(np.c_[baseunit, data[:, 0]], axis=1)
        order3_x1y2 = np.prod(np.c_[baseunit, data[:, 1]], axis=1)
        order4_x3y1 = np.prod(np.c_[order3_x2y1, data[:, 0]], axis=1)
        order4_x2y2 = np.prod(np.c_[order3_x2y1, data[:, 1]], axis=1)
        order4_x1y3 = np.prod(np.c_[order3_x1y2, data[:, 1]], axis=1)
        order5_x4y1 = np.prod(np.c_[order4_x3y1, data[:, 0]], axis=1)
        order5_x3y2 = np.prod(np.c_[order4_x3y1, data[:, 1]], axis=1)
        order5_x2y3 = np.prod(np.c_[order4_x1y3, data[:, 0]], axis=1)
        order5_x1y4 = np.prod(np.c_[order4_x1y3, data[:, 1]], axis=1)
        order6_x5y1 = np.prod(np.c_[order5_x4y1, data[:, 0]], axis=1)
        order6_x4y2 = np.prod(np.c_[order5_x4y1, data[:, 1]], axis=1)
        order6_x3y3 = np.prod(np.c_[order5_x2y3, data[:, 0]], axis=1)
        order6_x2y4 = np.prod(np.c_[order5_x2y3, data[:, 1]], axis=1)
        order6_x1y5 = np.prod(np.c_[order5_x1y4, data[:, 1]], axis=1)
        order7_x6y1 = np.prod(np.c_[order6_x5y1, data[:, 0]], axis=1)
        order7_x5y2 = np.prod(np.c_[order6_x5y1, data[:, 1]], axis=1)
        order7_x4y3 = np.prod(np.c_[order6_x3y3, data[:, 0]], axis=1)
        order7_x3y4 = np.prod(np.c_[order6_x3y3, data[:, 1]], axis=1)
        order7_x2y5 = np.prod(np.c_[order6_x1y5, data[:, 0]], axis=1)
        order7_x1y6 = np.prod(np.c_[order6_x1y5, data[:, 1]], axis=1)

        A = np.c_[np.ones(data.shape[0]), baseunit, order2_x1y1, data[:, :2] ** 2, order3_x2y1, order3_x1y2,
                  data[:, :2] ** 3, order4_x3y1, order4_x2y2, order4_x1y3, data[:, :2] ** 4, order5_x4y1, order5_x3y2,
                  order5_x2y3, order5_x1y4, data[:, :2] ** 5, order6_x5y1, order6_x4y2, order6_x3y3, order6_x2y4,
                  order6_x1y5, data[:, :2] ** 6, order7_x6y1, order7_x5y2, order7_x4y3, order7_x3y4, order7_x2y5,
                  order7_x1y6, data[:, :2] ** 7]

    A_end_col = int((order + 2) * (order + 1) / 2)
    C, residuals, _, _ = sp.linalg.lstsq(A[:, :A_end_col], data[:, 2])

    return C, residuals, A


def clean_surface(data, C):
    """
    input found surface, calculate distance of each point from surface, look at distribution of dist and remove outliers
    :param data: ndarray {N,3}
    :param C: coefficient list C00, C11, C12, C21,....
    :return: data_cln ndarray {N,3}
    19.4.20 added function from cushions_straight (previously it was missing) + some syntax changes: med/stdev for
    distance calculated once, and not again after first removal
    """

    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]
    calc_Z = np.dot(np.c_[np.ones(data.shape[0]), X, Y], C[:3])

    if C.shape[0] >= 6:
        calc_Z_2 = np.dot(np.c_[X * Y, X ** 2, Y ** 2], C[3:6])
        calc_Z = calc_Z + calc_Z_2

        if C.shape[0] >= 10:
            calc_Z_3 = np.dot(
                np.c_[X ** 2 * Y, X * Y ** 2, X ** 3, Y ** 3], C[6:10])
            calc_Z = calc_Z + calc_Z_3

            if C.shape[0] >= 15:
                calc_Z_4 = np.dot(np.c_[X ** 3 * Y, X ** 2 * Y ** 2, X * Y ** 3, X ** 4, Y ** 4], C[10:15])
                calc_Z = calc_Z + calc_Z_4

                if C.shape[0] >= 21:
                    calc_Z_5 = np.dot(np.c_[X ** 4 * Y, X ** 3 * Y ** 2, X ** 2 * Y ** 3, X * Y ** 4, X ** 5, Y ** 5],
                                      C[15:21])
                    calc_Z = calc_Z + calc_Z_5

                    if C.shape[0] >= 28:
                        calc_Z_6 = np.dot(np.c_[X ** 5 * Y, X ** 4 * Y ** 2, X ** 3 * Y ** 3, X ** 2 * Y ** 4,
                                                X ** 1 * Y ** 5, X ** 6, Y ** 6], C[21:28])
                        calc_Z = calc_Z + calc_Z_6

                        if C.shape[0] >= 36:
                            calc_Z_7 = np.dot(np.c_[X ** 6 * Y, X ** 5 * Y ** 2, X ** 4 * Y ** 3, X ** 3 * Y ** 4,
                                                    X ** 2 * Y ** 5, X ** 1 * Y ** 6, X ** 7, Y ** 7], C[28:36])
                            calc_Z = calc_Z + calc_Z_7

    dst = calc_Z - Z
    data_cln = np.copy(data)
    dst_cln = np.copy(dst)

    med_dst_cln = np.median(dst_cln)
    std_dst_cln = np.std(dst_cln)

    data_cln = data_cln[np.where(dst_cln > (med_dst_cln - 3 * std_dst_cln))]
    dst_cln = dst_cln[np.where(dst_cln > (med_dst_cln - 3 * std_dst_cln))]
    data_cln = data_cln[np.where(dst_cln < (med_dst_cln + 3 * std_dst_cln))]


    return data_cln


def fit_clean_surf(data, order=1, iter=5, coeff_mx_change=0.01):
    """
    Take input data, clean it and find best fit surface. Data cleaning --> fitting continues until max #
    iterations reached OR until there are no more outliers to be removed OR until pre-post difference between fit
    coefficients is less than threshold. Nested within: fit_surface, clean_surface

    :param data: ndarray {N,3}
    :param order: int between 1-7. Input higher than 7 will return 7th order fit
    :param iter: int max # iterations outlier removal
    :param coeff_mx_change: min difference between coefficients pre/post data cleaning to finalize params
    :return: data: clean data (N-X,3), , C: fit coefficients {M,1} (M: order-dependent), X, Y, Z: surface XYZ values
    (for plotting),r^2 of fit, s: string with fit equation (for plotting)
    """

    C, res, A = fit_surface(data, order)

    for i in range(iter):
        data_clean = clean_surface(data, C)
        if data_clean.shape[0] == data.shape[0]:
            break
        else:
            data = data_clean
            C_prev = C
            C, res, _ = fit_surface(data, order)
            if all(np.abs(C - C_prev) < coeff_mx_change * np.abs(C)):
                break

    return data, C, res


def fit_auto_order(data, max_diff=0.08, max_diff_factor=0.5):
    """
    Fit datapoints contained within selected point list to polynomial. Order of fit is determined based on variance
    calculation which takes into account squared residuals, # datapoints, # polynomial coefficients. Order of fit is
    1-7, and is determined by the point at which additional order increase does not improve variance by more than max_diff

    :param data: input clean point cloud [N,3]
    :param selected_point_list: input np array [N,2] of user selected points
    :param max_diff: float
    :return: C, X, Y, Z
    C: coefficient list. length is order dependent.
    order --> C.shape[0]:
    1 --> 3
    2 --> 6
    3 --> 10
    4 --> 15
    5 --> 21
    6 --> 28
    7 --> 36
    X,Y,Z. Dimensions of X,Y,Z: [# incr Y, # incr X]
    """

    n = data.shape[0]

    C_0, res_0, A = fit_surface(data, 1)
    k_0 = C_0.shape[0] - 1
    var_0 = res_0 / (n + k_0)

    for i in range(2, 8):
        C, res, _ = fit_surface(data, i, A)

        k_1 = C.shape[0] - 1
        var = res / (n + k_1)

        compare_orders = abs(var - var_0) / var_0

        if compare_orders < (max_diff if i==2 else max_diff_factor):
            C = C_0
            break
        elif i == 7:
            break
        else:
            C_0, var_0 = C, var

        # max_diff = max_diff * max_diff_factor

    return C, i


def meshgrid_x_y(data, selected_point_list=None, board_height=None, res_X_Y=0.001, rect_list=None):
    """
    Input data, selected point list (if exists), board height (if exists), increments of X,Y mesh grid (default 1mm),
    return X,Y as ndarray dimensions (range(min_X_input, max_X_input, res_X_Y), range(min_Y_input, max_Y_input, res_X_Y))
    :param data:
    :param selected_point_list:
    :param board_height:
    :param res_X_Y:
    :return: X,Y: ndarrays dimensions mxn: m=range(min_X_input, max_X_input, res_X_Y), n=range(min_Y_input, max_Y_input, res_X_Y)
    """
    if board_height:
        board_height_m = round(board_height / 1000, 3)  # convert board size mm --> m

    # find min, max X,Y from input data, round to 3rd # after decimal (=1mm)
    if selected_point_list is None:
        min_Y_input = round(min(data[:, 1]), 3)
        max_Y_input = round(max(data[:, 1]), 3)

        if board_height is None:
            min_X_input = round(min(data[:, 0]), 3)
            max_X_input = round(max(data[:, 0]), 3)
        else:
            min_X_input = round(0.5 * (max(data[:, 0]) + min(data[:, 0]) - board_height_m), 3)
            max_X_input = round(0.5 * (max(data[:, 0]) + min(data[:, 0]) + board_height_m), 3)

    else:
        min_Y_input = round(np.min(selected_point_list[:, 1]), 3)
        max_Y_input = round(np.max(selected_point_list[:, 1]), 3)

        if board_height is None:
            min_X_input = round(np.min(selected_point_list[:, 0]), 3)
            max_X_input = round(np.max(selected_point_list[:, 0]), 3)
        else:
            min_X_input = round(
                0.5 * (np.min(selected_point_list[:, 0]) + np.max(selected_point_list[:, 0]) - board_height_m),
                3)
            max_X_input = round(
                0.5 * (np.min(selected_point_list[:, 0]) + np.max(selected_point_list[:, 0]) + board_height_m),
                3)

    if rect_list is not None:
        rect_list_np = np.array([np.array(i) for i in rect_list])
        rect_list_min_Y = np.min(rect_list_np[:,2:])
        rect_list_max_Y = np.max(rect_list_np[:, 2:])

        if rect_list_min_Y < min_Y_input:
            min_Y_input = rect_list_min_Y
        if rect_list_max_Y > max_Y_input:
            max_Y_input = rect_list_max_Y

            # Grid for X,Y covering min-max X,Y 1mm increments
    X, Y = np.meshgrid(np.arange(min_X_input, max_X_input, res_X_Y),
                       np.arange(min_Y_input, max_Y_input, res_X_Y))

    return X, Y


def calc_z(C, X, Y):
    XX = X.flatten()
    YY = Y.flatten()

    axes = np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2, XX ** 2 * YY, XX * YY ** 2, XX ** 3, YY ** 3,
                 (XX ** 3) * YY, (XX ** 2) * (YY ** 2), XX * (YY ** 3), XX ** 4, YY ** 4, XX ** 4 * YY,
                 XX ** 3 * YY ** 2, XX ** 2 * YY ** 3, XX * YY ** 4, XX ** 5, YY ** 5, XX ** 5 * YY, XX ** 4 * YY ** 2,
                 XX ** 3 * YY ** 3, XX ** 2 * YY ** 4, XX * YY ** 5, XX ** 6, YY ** 6, XX ** 6 * YY, XX ** 5 * YY ** 2,
                 XX ** 4 * YY ** 3, XX ** 3 * YY ** 4, XX ** 2 * YY ** 5, XX * YY ** 6, XX ** 7, YY ** 7]

    Z = np.dot(axes[:, :C.shape[0]], C).reshape(X.shape)

    return Z


# def calc_z_minmax_sym(C, X, Y):
#     # find (x,y) for 4 corners:
#     x_0 = X[0][0]
#     x_1 = X[0][-1]
#     y_0 = Y[0][0]
#     y_1 = Y[-1][0]
#
#     x, y = sym.symbols('x y')
#     pwrs = [1, x, y, x*y, x**2, y**2, x**2 * y, x * y**2, x**3, y**3, x**3 * y, x**2 * y**2, x * y**3, x**4, y**4,
#             x**4 * y, x**3 * y**2, x**2 * y**3, x * y**4, x**5, y**5, x**5 * y, x**4 * y**2, x**3 * y**3, x**2 * y**4,
#             x * y**5, x**6, y**6, x**6 * y, x**5 * y**2, x**4 * y**3, x**3 * y**4, x**2 * y**5, x ** y**6, x**7, y**7]
#
#     z = 0
#     for i in range(C.shape[0]):
#         z = z + C[i] * pwrs[i]
#
#     z_00 = sym.solveset(z, x, domain=S.Reals)
#     z_x = sym.diff(z, x)
#     z_y = sym.diff(z, y)
#     z_xx = sym.diff(z, x, 2)
#     z_yy = sym.diff(z, y, 2)
#     z_xy = sym.diff(z, x, y)


def fit_Z_to_rectangles(data, C, rect_list, avg_high_perc, selected_point_list=None, board_height=None, X=None, Y=None, res_X_Y=0.001):
    """
    Input data/coefficients from fit_surface, list of rectangles. Each rectangle defined by 2 corners X0, X1, Y0, Y1
    Returns mean, min, max of Z from each rectangle
    6/4/20 new input: avg_high_perc: float 0-1. Percent of "high" (close to camera so low values) Z subset vals to be taken for high_avg
    :param Z:
    :param rect_list: list of lists: [[X00, X01, Y00, Y01],...,[Xn0, Xn1, Yn0, Yn1]]
    :param avg_high_perc: list of floats, len = len of rect_list
    :return:
    list of lists: [[Z0min, Z0max, Z0mean, Z0high_mean],...,[ZNmin, ZNmax, ZNmean, ZNhigh_mean]]
    19.4.20 change expected input avg_high_perc from float --> list of floats
    """

    # Grid for X,Y covering min-max X,Y 1mm increments
    if not X or not Y:
        X, Y = meshgrid_x_y(data, selected_point_list, board_height, res_X_Y, rect_list)

    Z_mean_min_max = []

    for row in range(len(rect_list)):
        X_0 = rect_list[row][0]
        X_1 = rect_list[row][1]
        Y_0 = rect_list[row][2]
        Y_1 = rect_list[row][3]

        if (X_0 >= 0 and X_1 >= 0) or (X_0 < 0 and X_1 < 0):
            X_rect = sorted([X_0, X_1])
        else:
            X_rect = sorted([X_0, X_1], reverse=True)

        if (Y_0 >= 0 and Y_1 >= 0) or (Y_0 < 0 and Y_1 < 0):
            Y_rect = sorted([Y_0, Y_1])
        else:
            Y_rect = sorted([Y_0, Y_1], reverse=True)

        X_0 = X_rect[0]
        X_1 = X_rect[1]
        Y_0 = Y_rect[0]
        Y_1 = Y_rect[1]

        X_sub = X[Y_0:Y_1 + 1, X_0:X_1 + 1]
        Y_sub = Y[Y_0:Y_1 + 1, X_0:X_1 + 1]
        Z_sub = calc_z(C, X_sub, Y_sub)

        Z_mean = np.mean(Z_sub)
        Z_min = np.min(Z_sub)
        Z_max = np.max(Z_sub)
        Z_high_mean = np.mean(np.sort(Z_sub, axis=None)[:int(avg_high_perc[row] * Z_sub.size)])
        Z_vector = [Z_min, Z_max, Z_mean, Z_high_mean]
        Z_vector = [Z_mean, Z_min , Z_max , Z_high_mean]
        Z_mean_min_max.append(Z_vector)

    return Z_mean_min_max


def board_warp(data, C, d1row=4, d2row=205, d1col=4, d2col=205, selected_point_list=None, board_height=None, res_X_Y=0.003):
    frameMinIsGlobalMin = False
    frameMaxIsGlobalMax = False
    warp_type = 0

    frame_inc = round(res_X_Y / 0.001)
    if frame_inc != 1:
        d1row = round(d1row / frame_inc)
        d2row = round(d2row / frame_inc)
        d1col = round(d1col / frame_inc)
        d2col = round(d2col / frame_inc)

    X, Y = meshgrid_x_y(data, selected_point_list, board_height, res_X_Y)
    Z = calc_z(C, X, Y)

    Z_frame = np.empty(shape=(Z.shape[0], Z.shape[1]))
    Z_frame[:] = np.NaN
    Z_frame[d1row: d2row, d2col: -d2col] = Z[d1row: d2row, d2col: -d2col]
    Z_frame[-d2row: -d1row, d2col: -d2col] = Z[-d2row: -d1row, d2col: -d2col]
    Z_frame[d1row: -d1row, d1col: d2col] = Z[d1row: -d1row, d1col: d2col]
    Z_frame[d1row: -d1row, -d2col: -d1col] = Z[d1row: -d1row, -d2col: -d1col]

    Z_frame_min = np.nanmin(Z_frame)
    Z_frame_max = np.nanmax(Z_frame)

    XY_frame_min = np.where(Z_frame == Z_frame_min)
    XY_frame_max = np.where(Z_frame == Z_frame_max)

    Z_global_min = np.min(Z)
    Z_global_max = np.max(Z)

    XY_global_min = np.where(Z == Z_global_min)
    XY_global_max = np.where(Z == Z_global_max)

    w1_Z = abs(Z_frame_max - Z_global_min)
    w2_Z = abs(Z_frame_min - Z_global_max)

    w1_XY = frame_inc * distance_matrix(np.transpose(XY_frame_max), np.transpose(XY_global_min))
    w2_XY = frame_inc * distance_matrix(np.transpose(XY_frame_min), np.transpose(XY_global_max))

    w1_mat = w1_Z / w1_XY
    w2_mat = w2_Z / w2_XY

    w1 = np.max(w1_mat)
    w2 = np.max(w2_mat)

    if w1 > w2:
        w = w1
        warp_type = 1
    elif w2 > w1:
        w = w2
        warp_type = 2
    else:
        w = w1

    if Z_global_min == Z_frame_min:
        frameMinIsGlobalMin = True

    if Z_global_max == Z_frame_max:
        frameMaxIsGlobalMax = True

    return w, warp_type, frameMinIsGlobalMin, frameMaxIsGlobalMax
