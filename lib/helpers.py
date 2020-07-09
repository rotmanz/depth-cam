import numpy as np
from skimage.measure import label,regionprops
import cv2
import matplotlib.pyplot as plt


def convert_depth_frame_to_pointcloud(depth_image , camera_intrinsics, depth_scale):
    """
    Convert the depthmap to a 3D point cloud
    Parameters:
    -----------
    depth_frame 	 	 : rs.frame()
                           The depth_frame containing the depth map
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
    Return:
    ----------
    x : array
        The x values of the pointcloud in meters
    y : array
        The y values of the pointcloud in meters
    z : array
        The z values of the pointcloud in meters
    """

    [height , width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    z = depth_image.flatten() * depth_scale
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    # x = x[np.nonzero(z)]
    # y = y[np.nonzero(z)]
    # z = z[np.nonzero(z)]

    x = x.reshape(depth_image.shape)
    y = y.reshape(depth_image.shape)
    z = z.reshape(depth_image.shape)

    return x, y, z


# !!!!!!!!!!!!!!!!!!!!!! DEPRECATED !!!!!!!!!!!!!!!!!!!!!!!!
# def rotatePointCloud(pointCloud,angle,axis):
#
#     if axis==1 or axis==0:
#         vtag = pointCloud[:,axis] * math.cos(angle) + pointCloud[:,2] * math.sin(angle)
#         ztag = -pointCloud[:,axis] * math.sin(angle) + pointCloud[:,2] * math.cos(angle)
#         pointCloud[:,axis]=vtag
#         pointCloud[:,2]=ztag
#     return pointCloud


def getLargestCC(segmentation):
    labels = label(segmentation)
    # assert (labels.max() != 0)  # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    else:
        largestCC = labels == 0
    return largestCC

def findMedia(irImage, colorImage , use_laser = True, isBrown = False, zImage=None, rescaleGrabcut = 1):

    irImage = np.copy(irImage)
    colorImage = np.copy(colorImage)

    stacked_img = np.stack((irImage ,) * 3 , axis=-1).astype(np.uint8)
    mask = np.ones(stacked_img.shape[:2] , np.uint8)*2

    bgdModel = np.zeros((1 , 65) , np.float64)
    fgdModel = np.zeros((1 , 65) , np.float64)

    # rect = (0 , 10 , 640 , 470)
    # cv2.grabCut(stacked_img , mask , rect , bgdModel , fgdModel , 5 , cv2.GC_INIT_WITH_RECT)
    mask_from_color = np.zeros(stacked_img.shape[:2] , np.uint8)
    if isBrown:
        colorImage = colorImage[:,:,0:2]

    tempImage = np.std(colorImage , axis=2) / np.mean(colorImage , axis=2)
    tempImage[np.mean(colorImage , axis=2) == 0] = 255
    tempImage2 = np.max(colorImage,axis=2)-np.min(colorImage,axis=2)
    mask_from_color[np.logical_and(tempImage < 0.2,tempImage2<50)] = 1
    mask_not_from_color=tempImage2>50

    # mask_from_color[tempImage2 < 50] = 1

    kernel = np.ones((5 , 5) , np.uint8)
    mask_from_color = cv2.morphologyEx(mask_from_color , cv2.MORPH_OPEN , kernel)

    mask_from_color = getLargestCC(mask_from_color)

    labels = label(irImage>100)
    regions = regionprops(labels)

    if use_laser:
        mask_from_laaser = np.zeros_like(mask,dtype=np.uint8)

        for props in regions:
            if isBrown:
                if props.area<20 and props.area>1:
                    mask_from_laaser[props.coords[:,0],props.coords[:,1]]=1
            else:
                if props.area<20 and props.area>1 and props.eccentricity<0.4:
                    mask_from_laaser[props.coords[:,0],props.coords[:,1]]=1

        mask_from_laaser = cv2.dilate(mask_from_laaser.astype(np.uint8) , np.ones((3,3),np.uint8) , iterations=1)==1

        mask[mask_from_laaser]=1

    mask_from_z = np.zeros(stacked_img.shape[:2] , np.uint8)
    if zImage is not None:
        zImage = np.copy(zImage)
        zImage[zImage > 2] = float('nan')
        zImage[zImage < 0.2] = float('nan')
        mask_from_z[np.isnan(zImage)] = 1
        # mask_from_z[zImage < 0.2] = 1
        normalizedImg = np.zeros_like(zImage , dtype=np.uint8)
        normalizedImg = cv2.normalize(zImage , normalizedImg , 0 , 255 , cv2.NORM_MINMAX)
        edges = cv2.Canny(normalizedImg.astype(np.uint8) , 20 , 100)
        mask_from_z[edges==255] = 1
        mask_from_z = cv2.morphologyEx(mask_from_z.astype(np.uint8) , cv2.MORPH_CLOSE , np.ones((7,7),np.uint8))
        sobely = cv2.Sobel(zImage,cv2.CV_8U,0,1,ksize=5)
        mask_from_z[sobely>0] = 1

    useColorGradient = True
    mask_from_gradient = np.zeros(stacked_img.shape[:2] , np.uint8)
    zeroColor = np.zeros(stacked_img.shape[:2] , np.uint8)
    if useColorGradient:
        if isBrown:
            grayImage = np.mean(colorImage,axis=2).astype(np.uint8)
        else:
            grayImage = cv2.cvtColor(colorImage.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # normalizedImg = np.zeros_like(grayImage , dtype=np.uint8)
        # normalizedImg = cv2.normalize(grayImage , normalizedImg , 0 , 255 , cv2.NORM_MINMAX)
        edges = cv2.Canny(grayImage.astype(np.uint8) , 20 , 100)
        mask_from_gradient[edges==255] = 1
        zeroColor [np.mean(colorImage , axis=2) == 0 ] = 1
        zeroColor = cv2.dilate(zeroColor , np.ones((3 ,3) , np.uint8))
        mask_from_gradient[zeroColor == 1] = 0
        mask_from_gradient = cv2.morphologyEx(mask_from_gradient , cv2.MORPH_CLOSE , np.ones((13,13),np.uint8))
        # mask_from_gradient [zeroColor==1] = 0


    mask[120:170 , 400:450] = 1
    # if not isBrown:
    mask[mask_from_color == 1] = 1
    # mask[np.logical_and(np.mean(colorImage , axis=2) > 0,mask_from_color == 0)] = 0
    mask[np.logical_and(np.mean(colorImage , axis=2) > 0 , mask_not_from_color)] = 0
    mask[10:160 , 615:639] = 0
    mask[cv2.morphologyEx((irImage==255).astype(np.uint8) , cv2.MORPH_OPEN , np.ones((3,3),np.uint8))==1]=0
    mask[:10 , :] = 0
    if zImage is not None:
        mask[mask_from_z==1] = 0

    if useColorGradient:
        mask[mask_from_gradient == 1] = 0

    if rescaleGrabcut > 1:
        stacked_img = cv2.resize(stacked_img , None , fx=1/rescaleGrabcut , fy=1/rescaleGrabcut , interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask , None , fx=1 / rescaleGrabcut , fy=1 / rescaleGrabcut ,
                                 interpolation=cv2.INTER_NEAREST)

    if np.sum(mask==1)>0:
        mask , bgdModel , fgdModel = cv2.grabCut(stacked_img , mask , None , bgdModel , fgdModel , 5 ,
                                                 cv2.GC_INIT_WITH_MASK)

    if rescaleGrabcut > 1:
        mask = cv2.resize(mask , None , fx=rescaleGrabcut , fy=rescaleGrabcut ,
                                 interpolation=cv2.INTER_NEAREST)

    mask2 = np.where((mask == 2) | (mask == 0) , 0 , 1).astype('uint8')
    mask2 = getLargestCC(mask2)

    return mask2