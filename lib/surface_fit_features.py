import numpy as np
import scipy
from scipy.spatial import distance_matrix
def fit_surface(data, order=1, selected_point_list=None, board_height=None, X=None, Y=None, axes=None, A=None):
    """
    input data: clean point cloud
    do 1st order fit (will support orders 1-7)
    return C, X, Y, Z, residuals
    return C: coefficient list. length is order dependent.
    order --> C.shape[0]:
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

    if axes is None:
        if X is None or Y is None:
            res_X_Y = 0.001  # mesh with 1mm jumps in X,Y
            if board_height:  # 010420 - Ziv - added if statements to support None as board_height
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
            # Grid for X,Y covering min-max X,Y 1mm increments
            X, Y = np.meshgrid(np.arange(min_X_input, max_X_input, res_X_Y,dtype=np.float32),
                               np.arange(min_Y_input, max_Y_input, res_X_Y,dtype=np.float32))
        XX = X.flatten()
        YY = Y.flatten()

        axes = np.c_[np.ones(XX.shape,dtype=np.float32), XX, YY, XX * YY, XX ** 2, YY ** 2, XX ** 2 * YY, XX * YY ** 2, XX ** 3, YY ** 3,
                     (XX ** 3) * YY, (XX ** 2) * (YY ** 2), XX * (YY ** 3), XX ** 4, YY ** 4, XX ** 4 * YY,
                     XX ** 3 * YY ** 2, XX ** 2 * YY ** 3, XX * YY ** 4, XX ** 5, YY ** 5, XX ** 5 * YY, XX ** 4 * YY ** 2,
                     XX ** 3 * YY ** 3, XX ** 2 * YY ** 4, XX * YY ** 5, XX ** 6, YY ** 6, XX ** 6 * YY, XX ** 5 * YY ** 2,
                     XX ** 4 * YY ** 3, XX ** 3 * YY ** 4, XX ** 2 * YY ** 5, XX * YY ** 6, XX ** 7, YY ** 7]

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

        A = np.c_[np.ones(data.shape[0],dtype = np.float32), baseunit, order2_x1y1, data[:, :2] ** 2, order3_x2y1, order3_x1y2,
                  data[:, :2] ** 3, order4_x3y1, order4_x2y2, order4_x1y3, data[:, :2] ** 4, order5_x4y1, order5_x3y2,
                  order5_x2y3, order5_x1y4, data[:, :2] ** 5, order6_x5y1, order6_x4y2, order6_x3y3, order6_x2y4,
                  order6_x1y5, data[:, :2] ** 6, order7_x6y1, order7_x5y2, order7_x4y3, order7_x3y4, order7_x2y5,
                  order7_x1y6, data[:, :2] ** 7]

    if order == 1:
        C, residuals, _, _ = scipy.linalg.lstsq(A[:, :3], data[:, 2])
        Z = np.dot(axes[:, :3], C).reshape(X.shape)
    elif order == 2:
        C, residuals, _, _ = scipy.linalg.lstsq(A[:, :6], data[:, 2])
        Z = np.dot(axes[:, :6], C).reshape(X.shape)
    elif order == 3:
        C, residuals, _, _ = scipy.linalg.lstsq(A[:, :10], data[:, 2])
        Z = np.dot(axes[:, :10], C).reshape(X.shape)
    elif order == 4:
        C, residuals, _, _ = scipy.linalg.lstsq(A[:, :15], data[:, 2])
        Z = np.dot(axes[:, :15], C).reshape(X.shape)
    elif order == 5:
        C, residuals, _, _ = scipy.linalg.lstsq(A[:, :21], data[:, 2])
        Z = np.dot(axes[:, :21], C).reshape(X.shape)
    elif order == 6:
        C, residuals, _, _ = scipy.linalg.lstsq(A[:, :28], data[:, 2])
        Z = np.dot(axes[:, :28], C).reshape(X.shape)
    elif order == 7:
        C, residuals, _, _ = scipy.linalg.lstsq(A, data[:, 2])
        Z = np.dot(axes, C).reshape(X.shape)

    return C, X, Y, Z, residuals, A, axes


def clean_surface(data, C):
    """
    input found surface, calculate distance of each point from surface, look at distribution of dist and remove outliers
    :param data: ndarray {N,3}
    :param C: coefficient list C00, C11, C12, C21,....
    :return: data_cln ndarray {N,3}
    19.4.20 added function from cushions_straight (previously it was missing) + some syntax changes: med/stdev for
    distance calculated once, and not again after first removal
    """

    calc_Z = np.dot(np.c_[np.ones(data.shape[0]), data[:, 0], data[:, 1]], C[:3])

    if C.shape[0] >= 6:
        calc_Z_2 = np.dot(np.c_[data[:, 0] * data[:, 1], data[:, 0] ** 2, data[:, 1] ** 2], C[3:6])
        calc_Z = calc_Z + calc_Z_2

        if C.shape[0] >= 10:
            calc_Z_3 = np.dot(
                np.c_[data[:, 0] ** 2 * data[:, 1], data[:, 0] * data[:, 1] ** 2, data[:, 0] ** 3, data[:, 1] ** 3],
                C[6:10])
            calc_Z = calc_Z + calc_Z_3

            if C.shape[0] >= 15:
                calc_Z_4 = np.dot(np.c_[data[:, 0] ** 3 * data[:, 1], data[:, 0] ** 2 * data[:, 1] ** 2, data[:, 0] *
                                        data[:, 1] ** 3, data[:, 0] ** 4, data[:, 1] ** 4], C[10:15])
                calc_Z = calc_Z + calc_Z_4

                if C.shape[0] >= 21:
                    calc_Z_5 = np.dot(np.c_[data[:, 0] ** 4 * data[:, 1], data[:, 0] ** 3 * data[:, 1] ** 2,
                                            data[:, 0] ** 2 * data[:, 1] ** 3, data[:, 0] * data[:, 1] ** 4,
                                            data[:, 0] ** 5, data[:, 1] ** 5], C[15:21])
                    calc_Z = calc_Z + calc_Z_4

    dst = calc_Z - data[:, 2]
    data_cln = np.copy(data)
    dst_cln = np.copy(dst)

    med_dst_cln = np.median(dst_cln)
    std_dst_cln = np.std(dst_cln)

    data_cln = data_cln[np.where(dst_cln > (med_dst_cln - 3 * std_dst_cln))]
    dst_cln = dst_cln[np.where(dst_cln > (med_dst_cln - 3 * std_dst_cln))]
    data_cln = data_cln[np.where(dst_cln < (med_dst_cln + 3 * std_dst_cln))]
    # dst_cln = dst_cln[np.where(dst_cln < (med_dst_cln + 3 * std_dst_cln))] #Ziv 190420 - commented line as value is unused

    return data_cln


def fit_clean_surf(data, order=1, iter=5, coeff_mx_change=0.01, selected_point_list=None, board_height=None):
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

    C, X, Y, Z, res, A, axes = fit_surface(data, order, selected_point_list, board_height)

    for i in range(iter):
        data_clean = clean_surface(data, C)
        if data_clean.shape[0] == data.shape[0]:
            break
        else:
            data = data_clean
            C_prev = C
            # C, _, _, Z, res, _, _ = fit_surface(data, order, selected_point_list, board_height, X=X, Y =Y, axes=axes, A=A) #19.4.20 Tzvia: change to use X,Y,axes,A from previous Ziv: removed A since it can't be reused
            C , _ , _ , Z , res , _ , _ = fit_surface(data , order , selected_point_list , board_height , X=X , Y=Y ,
                                                      axes=axes)  # 19.4.20 Tzvia: change to use X,Y,axes,A from previous Ziv: removed A since it can't be reused
            if all(np.abs(C - C_prev) < coeff_mx_change * np.abs(C)):
                break

    return data, C, X, Y, Z, res


def fit_auto_order(data, selected_point_list=None, board_height=None, max_diff=0.08, max_diff_factor=0.5):
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

    C_0, X, Y, Z_0, res_0, A, axes = fit_surface(data, 1, selected_point_list, board_height) # Ziv - chnaged to use new fit surface method
    k_0 = C_0.shape[0] - 1
    var_0 = res_0 / (n + k_0)

    for i in range(2, 8):
        C, _, _, Z, res, _,_ = fit_surface(data, i, selected_point_list, board_height, X= X, Y = Y,axes = axes , A = A) # Ziv - chnaged to use new fit surface method

        k_1 = C.shape[0] - 1  # 010420 - Ziv - changed to k_1 from k (typo)
        var = res / (n + k_1)

        compare_orders = abs(var - var_0) / var_0

        if compare_orders < (max_diff if i==2 else max_diff_factor):
            C, Z = C_0, Z_0
            break
        elif i == 7:
            break
        else:
            C_0, Z_0, var_0 = C, Z, var

        # max_diff = max_diff * max_diff_factor

    return C, X, Y, Z, i  # 010420 - Ziv - added order for monitoring


def fit_Z_to_rectangles(Z, rect_list, avg_high_perc=None):
    """
    Input Z from fit_surface, list of rectangles. Each rectangle defined by 2 corners X0, X1, Y0, Y1
    Returns mean, min, max of Z from each rectangle
    6/4/20 new input: avg_high_perc: float 0-1. Percent of "high" (close to camera so low values) Z subset vals to be taken for high_avg
    :param Z:
    :param rect_list: list of lists: [[X00, X01, Y00, Y01],...,[Xn0, Xn1, Yn0, Yn1]]
    :param avg_high_perc: list of floats, len = len of rect_list
    :return:
    Old:
    list of lists: [[Z0mean, Z0min, Z0max],...,[Znmean, Znmin, Znmax]]
    6.4.20 change return: order of Z values + 4 vals per row instead of 3:
    list of lists: [[Z0min, Z0max, Z0mean, Z0high_mean],...,[ZNmin, ZNmax, ZNmean, ZNhigh_mean]]
    19.4.20 change expected input avg_high_perc from float --> list of floats

    """

    if avg_high_perc is None:
        avg_high_perc = np.ones(len(rect_list))
    else:
        avg_high_perc = avg_high_perc/100 #Ziv 030520 - added division by 100 to go from percents to 0..1

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

        Z_subset = Z[Y_0:Y_1 + 1, X_0:X_1 + 1]
        Z_mean = np.mean(Z_subset)
        Z_min = np.min(Z_subset)
        Z_max = np.max(Z_subset)
        Z_high_mean = np.mean(np.sort(Z_subset, axis=None)[:int(avg_high_perc[row] * Z_subset.size)])  # 6.4.20 Tzvia new "high mean" + 19.4.20 add [row] to avg_high_perc
        Z_vector = [Z_mean, Z_min, Z_max, Z_high_mean]  # 6.4.20 Tzvia change order of return + new output: mean,min,max--->min,max,mean,high mean
        Z_mean_min_max.append(Z_vector)


    return Z_mean_min_max


def board_warp(Z, d1row=4, d2row=205, d1col=4, d2col=205):
    frameMinIsGlobalMin = False
    frameMaxIsGlobalMax = False
    warp_type = 0

    Z_frame = np.empty(shape=(Z.shape[0], Z.shape[1]))
    Z_frame[:] = np.NaN
    Z_frame[d1row: d2row, d2col: -d2col] = Z[d1row: d2row, d2col: -d2col]
    Z_frame[-d2row: -d1row, d2col: -d2col] = Z[-d2row: -d1row, d2col: -d2col]
    Z_frame[d1row: -d1row, d1col: d2col] = Z[d1row: -d1row, d1col: d2col]
    Z_frame[d1row: -d1row, -d2col: -d1col] = Z[d1row: -d1row, -d2col: -d1col]

    Z_frame_min = np.nanmin(Z_frame)
    Z_frame_max = np.nanmax(Z_frame)

    XY_frame_min = np.where(Z_frame == Z_frame_min)
    # XY_frame_min_test = np.unravel_index(np.nanargmin(Z_frame), dims=(Z.shape[0], Z.shape[1])) # more efficient? only returns first incidence though...
    XY_frame_max = np.where(Z_frame == Z_frame_max)
    # XY_frame_max_test = np.unravel_index(np.nanargmax(Z_frame), dims=(Z.shape[0], Z.shape[1])) # more efficient? only returns first incidence though...

    Z_global_min = np.min(Z)
    Z_global_max = np.max(Z)

    XY_global_min = np.where(Z == Z_global_min)
    # XY_global_min = np.unravel_index(np.nanargmin(Z), dims=(Z.shape[0], Z.shape[1])) # more efficient? only returns first incidence though...
    XY_global_max = np.where(Z == Z_global_max)
    # XY_global_max = np.unravel_index(np.nanargmax(Z), dims=(Z.shape[0], Z.shape[1]))  # more efficient? only returns first incidence though...

    w1_Z = abs(Z_frame_max - Z_global_min)
    w2_Z = abs(Z_frame_min - Z_global_max)

    w1_XY = distance_matrix(np.transpose(XY_frame_max), np.transpose(XY_global_min))
    w2_XY = distance_matrix(np.transpose(XY_frame_min), np.transpose(XY_global_max))

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
