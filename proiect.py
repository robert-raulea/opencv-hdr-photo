import cv2 as cv
import numpy as np

#Load exposure images into a list
img_names = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
img_list = [cv.imread(fn) for fn in img_names]

exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

calibrateDebevec = cv.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(img_list, exposure_times)

mergeDebevec = cv.createMergeDebevec()
hdrDebevec = mergeDebevec.process(img_list, exposure_times, responseDebevec)

#Merge
# merge_debevec = cv.createMergeDebevec()
# hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())

# merge_robertson = cv.createMergeRobertson()
# hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())


# #Tonemap HDR image
# tonemap1 = cv.createTonemap(gamma=1)
# res_debevec = tonemap1.process(hdr_debevec.copy())

# #Merge exposures using Mertens fusion
# merge_mertens = cv.createMergeMertens()
# res_mertens = merge_mertens.process(img_list)

# # Convert datatype to 8-bit and save
# res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
# res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

cv.imwrite("hdrDebevec.jpg", hdrDebevec)
#cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)