import numpy as np
import cv2

pattern1 = (np.load("pattern/cifar100_preactresnet18_0_255.npy")).astype(np.uint8)
cv2.imshow("test", pattern1)
pattern2 = (np.load("save_folder/cifar10_preactresnet18_13.npy") * 255).astype(
    np.uint8
)[0]
cv2.imshow("test1", pattern2)
cv2.waitKey(0)
