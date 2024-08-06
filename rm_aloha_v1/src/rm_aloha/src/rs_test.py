import pyrealsense2 as rs
import numpy as np
import cv2 as cv

ctx = rs.context()
print(ctx.devices[0], '\n', ctx.devices[1], '\n', ctx.devices[2])
