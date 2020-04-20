# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob, os


def parse_file_name(img_path):
    user_id, _ = os.path.splitext(os.path.basename(img_path))
    #print(user_id)
    return np.array([user_id], dtype=np.uint16)

def parse_file_name2(img_path):
    user_id, _ = os.path.splitext(os.path.basename(img_path))
    user_id, finger_id = user_id.split('_')
    #print(user_id, finger_id)
    return np.array([user_id], dtype=np.uint16)

def hist_stretch(img_buf, width, height, shift):
    tmp1 = 0
    tmp2 = 0
    w_pHistBuf = np.zeros(256, dtype=np.uint32)
    total = 0
    ret_buf = np.zeros((width, height, 1), dtype=np.uint8)

    # get brightness
    for i in range(height):
        for j in range(width):
            total = total + img_buf[i][j][0]
            w_pHistBuf[img_buf[i][j][0]] += 1
    diff = (int)(shift - (total / (width * height)))
    w_pHistBuf = np.zeros(256, dtype=np.uint32)

    # move histogram
    for i in range(height):
        for j in range(width):
            tmp = img_buf[i][j][0] + diff
            if (tmp > 255):
                tmp = 255
            elif (tmp < 0):
                tmp = 0
            ret_buf[i][j][0] = tmp
            w_pHistBuf[tmp] += 1

    # stretch histogram
    for i in range(256):
        if (w_pHistBuf[i] != 0):
            tmp1 = i
            break
    for i in range(255, 0, -1):
        if (w_pHistBuf[i] != 0):
            tmp2 = i
            break
    for i in range(height):
        for j in range(width):
            ret_buf[i][j][0] = (int)((255 * (ret_buf[i][j][0] - tmp1) / (tmp2 - tmp1)))
    
    return ret_buf



# %%
# make data
img_list = sorted(glob.glob('../../dataset/train_data/*.bmp'))
print(len(img_list))

imgs = np.empty((len(img_list), 160, 160, 1), dtype = np.uint8)
labels = np.empty((len(img_list), 1), dtype = np.uint16)

for i, img_path in enumerate(img_list):
    print(i)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #tmp_img = img.reshape(160, 160, 1)
    #imgs[i] = hist_stretch(tmp_img, 160, 160, 128)
    imgs[i] = img.reshape(160, 160, 1)

    # get user id
    labels[i] = parse_file_name2(img_path)

np.save('../../dataset/np_data/img_train.npy', imgs)
np.save('../../dataset/np_data/label_train.npy', labels)



img_list = sorted(glob.glob('../../dataset/real_data/*.bmp'))
print(len(img_list))

imgs = np.empty((len(img_list), 160, 160, 1), dtype = np.uint8)
labels = np.empty((len(img_list), 1), dtype = np.uint16)

for i, img_path in enumerate(img_list):
    print(i)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #tmp_img = img.reshape(160, 160, 1)
    #imgs[i] = hist_stretch(tmp_img, 160, 160, 128)
    imgs[i] = img.reshape(160, 160, 1)

    # get user id
    labels[i] = parse_file_name(img_path)

np.save('../../dataset/np_data/img_real.npy', imgs)
np.save('../../dataset/np_data/label_real.npy', labels)

print('Complete')


# %%
