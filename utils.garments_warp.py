import cv2
import numpy as np

def warp_garment(garment, src_points, dst_points, size):
    matrix, _ = cv2.findHomography(src_points, dst_points)
    warped = cv2.warpPerspective(garment, matrix, size)
    return warped