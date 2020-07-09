# cushion application: option either to receive cushion correction (True/False/P/XP) as input OR determine whether to apply cushions and in what direction
import numpy as np
import scipy
import scipy.linalg

def cushion_adjustment_w_decision(data):
    """
    receives straightened data ndarray (N,3), does 1st order fit: print(camera coordinates: X) and x-print(camera coordinates: Y)
    Decision (placeholder logic) whether to apply cushions, and if so, print or x-print
    Returns data as it would be after applying cushions
    :param data: ndarray (N,3)
    :return: data_after_cush_corr: ndarray (N,3)
    """
    cushion_height = 40  # mm
    elev_xprint = 1320  # mm
    elev_print = 2500  # mm

    max_slope_xprint = cushion_height / elev_xprint
    max_slope_print = cushion_height / elev_print

    # coarse filtering of data
    data_coarse_filter = filter_edges_holes(data)

    # 1st order fit for cushions decision
    data_pre_cush, C_pre_cush, X_pre_cush, Y_pre_cush, Z_pre_cush, r2_pre_cush, s__pre_cush = fit_clean_surf(
        data_coarse_filter)

    apply_cushions_xprint = False
    apply_cushions_print = False
    data_post_cush = np.copy(data_pre_cush)
    C_cushion_corr = [0, 0, 0]

    if abs(C_pre_cush[1]) >= 0.8 * max_slope_xprint and C_pre_cush[1] > C_pre_cush[2]:
        # or some other logic TBD to determine whether cushions are used for xp/p/not at all...
        apply_cushions_xprint = True
        if np.sign(C_pre_cush[1]) == 1:
            C_cushion_corr = [0, max_slope_xprint, 0]
        else:
            C_cushion_corr = [0, -max_slope_xprint, 0]

    else:
        if C_pre_cush[2] >= 0.8 * max_slope_print:  # placeholder logic...
            apply_cushions_print = True
            if np.sign(C_pre_cush[2]) == 1:
                C_cushion_corr = [0, 0, max_slope_print]
            else:
                C_cushion_corr = [0, 0, -max_slope_print]

    if apply_cushions_xprint or apply_cushions_print:  # i.e. there is a cushion adjustment to be made
        data_post_cush = data_normalize(data_pre_cush, C_cushion_corr)

    return data_post_cush


def cushion_adjustment_wo_decision(data, apply_cushions_xprint=False, apply_cushions_print=False):
    """
    receives straightened data ndarray (N,3), does 1st order fit: print(camera coordinates: X) and x-print(camera coordinates: Y)
    Note that valid entries: False/False, False/True, True/False. Cannot be True/True. False/False will return input data.
    :param data: ndarray (N,3)
    :param apply_cushions_xprint: bool
    :param apply_cushions_print: bool
    :return: data_after_cush_corr: ndarray (N,3)
    """

    cushion_height = 40  # mm
    elev_xprint = 1320  # mm
    elev_print = 2500  # mm

    max_slope_xprint = cushion_height / elev_xprint
    max_slope_print = cushion_height / elev_print

    # coarse filtering of data
    data_coarse_filter = filter_edges_holes(data)

    # 1st order fit for cushions correction
    data_pre_cush, C_pre_cush, X_pre_cush, Y_pre_cush, Z_pre_cush, r2_pre_cush, s__pre_cush = fit_clean_surf(
        data_coarse_filter)

    data_post_cush = np.copy(data_pre_cush)
    C_cushion_corr = [0, 0, 0]
    if apply_cushions_xprint or apply_cushions_print:  # i.e. there is a cushion adjustment to be made
        if apply_cushions_xprint:
            if np.sign(C_pre_cush[1]) == 1:
                C_cushion_corr = [0, max_slope_xprint, 0]
            elif np.sign(C_pre_cush[1] == -1):
                C_cushion_corr = [0, -max_slope_xprint, 0]
        elif apply_cushions_print:
            if np.sign(C_pre_cush[2]) == 1:
                C_cushion_corr = [0, 0, max_slope_print]
            elif np.sign(C_pre_cush[2] == -1):
                C_cushion_corr = [0, 0, -max_slope_print]

        data_post_cush = data_normalize(data_pre_cush, C_cushion_corr)

    return data_post_cush



def filter_edges_holes(data):
    """
    Coarse data cleaning: remove datapoints where:
     1) X,Y or Z == 0 (is this relevant after data is straightened? Might want to remove...)
     X,Y or Z out of 3 sigma (calculation for med, stddev Z is done after X,Y cleaning)
    :param data: ndarray {N,3}
    :return: data_filtered: clean data: ndarray {N,3}
    """
    data_filtered = np.copy(data)

    #remove datapoints where one of axes == 0
    data_filtered = data_filtered[data_filtered[:, 0] != 0]
    data_filtered = data_filtered[data_filtered[:, 1] != 0]
    data_filtered = data_filtered[data_filtered[:, 2] != 0]

    medX = np.median(data[:, 0])
    medY = np.median(data[:, 1])
    stdevX = np.std(data[:, 0])
    stdevY = np.std(data[:, 1])

    data_filtered = data_filtered[data_filtered[:, 0] > (medX - 3 * stdevX)]
    data_filtered = data_filtered[data_filtered[:, 0] < (medX + 3 * stdevX)]

    data_filtered = data_filtered[data_filtered[:, 1] > (medY - 3 * stdevY)]
    data_filtered = data_filtered[data_filtered[:, 1] < (medY + 3 * stdevY)]

    medZ = np.median(data_filtered[:, 2])
    stdevZ = np.std(data_filtered[:, 2])

    data_filtered = data_filtered[data_filtered[:, 2] > (medZ - 3 * stdevZ)]
    data_filtered = data_filtered[data_filtered[:, 2] < (medZ + 3 * stdevZ)]

    return data_filtered


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

    C, X, Y, Z, r2, s = fit_surface(data, order)

    for i in range(iter):
        data_clean = clean_surface(data, C)
        if data_clean.shape[0] == data.shape[0]:
            break
        else:
            data = data_clean
            C_prev = C
            C, X, Y, Z, r2, s = fit_surface(data, order)
            if all(np.abs(C - C_prev) < coeff_mx_change * np.abs(C)):
                break

    return data, C, X, Y, Z, r2, s


def data_normalize(data, C_elev):
    """
    take data and correction vector [X, Y, Z]. Tilt data by vector
    :param data: ndarray {N,3}
    :param C_elev: list [1,3]
    :return: data_norm ndarray {N,3}
    """
    data_elevPlane = np.dot(np.c_[np.ones(data[:, 0].shape), data[:, 0], data[:, 1]], C_elev)
    data_norm = data
    data_norm[:, 2] = data[:, 2] - data_elevPlane
    return data_norm


def fit_surface(data, order=1):
    """
    Fit surface to 3d datapoints according to defined order
    :param data: ndarray {N,3}
    :param order: int 1-7
    :return: C: coefficient list {, X, Y, Z, r2, s
    """
    incr_axes = 10
    extrap_surf_fact = 1.00

    xMax = extrap_surf_fact * max(data[:, 0])
    xMin = extrap_surf_fact * min(data[:, 0])
    yMax = extrap_surf_fact * max(data[:, 1])
    yMin = extrap_surf_fact * min(data[:, 1])
    xInc = (xMax - xMin) / incr_axes
    yInc = (yMax - yMin) / incr_axes

    # regular grid covering the domain of the data
    X, Y = np.meshgrid(np.arange(xMin, xMax, xInc), np.arange(yMin, yMax, yInc))
    XX = X.flatten()
    YY = Y.flatten()
    baseunit = np.c_[data[:, :2]]

    if order > 0:
        order1_A = np.c_[np.ones(data.shape[0]), baseunit]
        A = order1_A  # A: (1, x, y)
        axes = np.c_[np.ones(XX.shape), XX, YY]

        if order == 1:
            C, residuals, _, _ = scipy.linalg.lstsq(A, data[:, 2])
            s = '{0:.3f} + {1:.3f}*x + {2:.3f}*y'.format(C[0], C[1], C[2])

        else:
            order2_x1y1 = np.prod(np.c_[baseunit], axis=1)
            order2_A = np.c_[order2_x1y1, data[:, :2] ** 2]
            order2_axes = np.c_[XX * YY, XX ** 2, YY ** 2]
            A = np.append(A, order2_A, axis=1)
            axes = np.c_[axes, order2_axes]
            if order == 2:
                C, residuals, _, _ = scipy.linalg.lstsq(A, data[:, 2])
                s = '{0:.3f} + {1:.3f}*x + {2:.3f}*y + {3:.3f}*x*y + {4:.3f}*$x^2$ + {5:.3f}*$y^2$' \
                    .format(C[0], C[1], C[2], C[3], C[4], C[5])
            else:
                order3_x2y1 = np.prod(np.c_[baseunit, data[:, 0]], axis=1)
                order3_x1y2 = np.prod(np.c_[baseunit, data[:, 1]], axis=1)
                order3_A = np.c_[order3_x2y1, order3_x1y2, data[:, :2] ** 3]
                order3_axes = np.c_[XX ** 2 * YY, XX * YY ** 2, XX ** 3, YY ** 3]
                A = np.append(A, order3_A, axis=1)
                axes = np.c_[axes, order3_axes]
                if order == 3:
                    C, residuals, _, _ = scipy.linalg.lstsq(A, data[:, 2])
                    s = '{0:.3f} + {1:.3f} * x + {2:.3f} * y + {3:.3f} * xy + {4:.3f} * $x^2$ + {5:.3f} * $y^2$ + ' \
                        '{6:.3f} * $x^2$y + {7:.3f} * x*$y^2$ + {8:.3f} * $x^3$ + {9:.3f} * $y^3$'. \
                        format(C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9])

                else:
                    order4_x3y1 = np.prod(np.c_[order3_x2y1, data[:, 0]], axis=1)
                    order4_x2y2 = np.prod(np.c_[order3_x2y1, data[:, 1]], axis=1)
                    order4_x1y3 = np.prod(np.c_[order3_x1y2, data[:, 1]], axis=1)
                    order4_A = np.c_[order4_x3y1, order4_x2y2, order4_x1y3, data[:, :2] ** 4]
                    order4_axes = np.c_[(XX ** 3) * YY, (XX ** 2) * (YY ** 2), XX * (YY ** 3), XX ** 4, YY ** 4]
                    A = np.append(A, order4_A, axis=1)
                    axes = np.c_[axes, order4_axes]
                    if order == 4:
                        C, residuals, _, _ = scipy.linalg.lstsq(A, data[:, 2])
                        s = '{0:.3f} + {1:.3f} * x + {2:.3f} * y + \n{3:.3f} * x*y + {4:.3f} * $x^2$ + {5:.3f} * $y^2$ + ' \
                            '\n{6:.3f} * $x^2$*y + {7:.3f} * x*$y^2$ + {8:.3f} * $x^3$ + {9:.3f} * $y^3$ + \n{10:.3f} * $x^3$*y + ' \
                            '{11:.3f} * $x^2*y^2$ + {12:.3f} * x*$y^3$ + {13:.3f} * $x^4$ + {14:.3f} * $y^4$'. \
                            format(C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9], C[10], C[11], C[12],
                                   C[13], C[14])
                    else:
                        order5_x4y1 = np.prod(np.c_[order4_x3y1, data[:, 0]], axis=1)
                        order5_x3y2 = np.prod(np.c_[order4_x3y1, data[:, 1]], axis=1)
                        order5_x2y3 = np.prod(np.c_[order4_x1y3, data[:, 0]], axis=1)
                        order5_x1y4 = np.prod(np.c_[order4_x1y3, data[:, 1]], axis=1)
                        order5_A = np.c_[order5_x4y1, order5_x3y2, order5_x2y3, order5_x1y4, data[:, :2] ** 5]
                        order5_axes = np.c_[
                            XX ** 4 * YY, XX ** 3 * YY ** 2, XX ** 2 * YY ** 3, XX * YY ** 4, XX ** 5, YY ** 5]
                        A = np.append(A, order5_A, axis=1)
                        axes = np.c_[axes, order5_axes]

                        if order == 5:
                            C, residuals, _, _ = scipy.linalg.lstsq(A, data[:, 2])
                            s = '{0:.3f} + {1:.3f} * x + {2:.3f} * y + \n{3:.3f} * x*y + {4:.3f} * $x^2$ + {5:.3f} * $y^2$ + ' \
                                '\n{6:.3f} * $x^2$*y + {7:.3f} * x*$y^2$ + {8:.3f} * $x^3$ + {9:.3f} * $y^3$ + \n{10:.3f} * $x^3$*y + ' \
                                '{11:.3f} * $x^2$*$y^2$ + {12:.3f} * x*$y^3$ + {13:.3f} * $x^4$ + {14:.3f} * $y^4$\n + {15:.3f} * $x^4$ * y' \
                                '+ {16:.3f} * $x^3$*$y^2$ + {17:.3f} * $x^2$*$y^3$ + {18:.3f} * x*$y^4$ + {19:.3f} * $x^5$ + {20:.3f} * $y^5$'. \
                                format(C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9], C[10], C[11], C[12],
                                       C[13], C[14], C[15], C[16], C[17], C[18], C[19], C[20])

                        else:
                            order6_x5y1 = np.prod(np.c_[order5_x4y1, data[:, 0]], axis=1)
                            order6_x4y2 = np.prod(np.c_[order5_x4y1, data[:, 1]], axis=1)
                            order6_x3y3 = np.prod(np.c_[order5_x2y3, data[:, 0]], axis=1)
                            order6_x2y4 = np.prod(np.c_[order5_x2y3, data[:, 1]], axis=1)
                            order6_x1y5 = np.prod(np.c_[order5_x1y4, data[:, 1]], axis=1)
                            order6_A = np.c_[
                                order6_x5y1, order6_x4y2, order6_x3y3, order6_x2y4, order6_x1y5, data[:, :2] ** 6]
                            order6_axes = np.c_[
                                XX ** 5 * YY, XX ** 4 * YY ** 2, XX ** 3 * YY ** 3, XX ** 2 * YY ** 4, XX * YY ** 5, XX ** 6, YY ** 6]
                            A = np.append(A, order6_A, axis=1)
                            axes = np.c_[axes, order6_axes]

                            if order == 6:
                                C, residuals, _, _ = scipy.linalg.lstsq(A, data[:, 2])
                                s = '6th order fit'

                            else:
                                order7_x6y1 = np.prod(np.c_[order6_x5y1, data[:, 0]], axis=1)
                                order7_x5y2 = np.prod(np.c_[order6_x5y1, data[:, 1]], axis=1)
                                order7_x4y3 = np.prod(np.c_[order6_x3y3, data[:, 0]], axis=1)
                                order7_x3y4 = np.prod(np.c_[order6_x3y3, data[:, 1]], axis=1)
                                order7_x2y5 = np.prod(np.c_[order6_x1y5, data[:, 0]], axis=1)
                                order7_x1y6 = np.prod(np.c_[order6_x1y5, data[:, 1]], axis=1)
                                order7_A = np.c_[
                                    order7_x6y1, order7_x5y2, order7_x4y3, order7_x3y4, order7_x2y5, order7_x1y6, data[
                                                                                                                  :,
                                                                                                                  :2] ** 7]
                                order7_axes = np.c_[
                                    XX ** 6 * YY, XX ** 5 * YY ** 2, XX ** 4 * YY ** 3, XX ** 3 * YY ** 4, XX ** 2 * YY ** 5, XX * YY ** 6, XX ** 7, YY ** 7]
                                A = np.append(A, order7_A, axis=1)
                                axes = np.c_[axes, order7_axes]

                                C, residuals, _, _ = scipy.linalg.lstsq(A, data[:, 2])
                                s = '7th order fit'

    Z = np.dot(axes, C).reshape(X.shape)
    r2 = 1 - residuals / (data[:, 2].size * data[:, 2].var())

    return C, X, Y, Z, r2, s


def clean_surface(data, C):
    """
    input found surface, calculate distance of each point from surface, look at distribution of dist and remove outliers
    :param data: ndarray {N,3}
    :param C: coefficient list C00, C11, C12, C21,....
    :return: data_cln ndarray {N,3}
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

    dstnce = calc_Z - data[:, 2]
    data_cln = data
    dstnce_cln = dstnce

    data_cln = data_cln[np.where(dstnce_cln > (np.median(dstnce_cln) - 3 * np.std(dstnce_cln)))]
    dstnce_cln = dstnce_cln[np.where(dstnce_cln > (np.median(dstnce_cln) - 3 * np.std(dstnce_cln)))]
    data_cln = data_cln[np.where(dstnce_cln < (np.median(dstnce_cln) + 3 * np.std(dstnce_cln)))]
    dstnce_cln = dstnce_cln[np.where(dstnce_cln < (np.median(dstnce_cln) + 3 * np.std(dstnce_cln)))]

    return data_cln


