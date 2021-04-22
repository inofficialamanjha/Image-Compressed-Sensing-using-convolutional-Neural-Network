from os import listdir
from os.path import join
import cv2
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', 'bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

idx = 0
for path in listdir('location/train'):
  path = 'location/train/' + path
  if is_image_file(path) == False:
    continue
  img = cv2.imread(path)
  x = 0
  y = 0
  while y < img.shape[0]:
    if x+96 < img.shape[1] and y+96 < img.shape[0]:
      img_org = img[y:y+96 , x:x+96]
      filename = 'location/Train_sub_images' + '/' + str(idx) + '.jpg'
      cv2.imwrite(filename , img_org)
      idx = idx+1
      img_flip = cv2.flip(img_org , 0)
      filename = 'location/Train_sub_images' + '/' + str(idx) + '.jpg'
      cv2.imwrite(filename , img_flip)
      idx = idx+1
      img_rotate_90 = rotate_image(img_org , 90)
      filename = 'location/Train_sub_images' + '/' + str(idx) + '.jpg'
      cv2.imwrite(filename , img_rotate_90)
      idx = idx+1
      img_rotate_90_flip = cv2.flip(img_rotate_90 , 0)
      filename = 'location/Train_sub_images' + '/' + str(idx) + '.jpg'
      cv2.imwrite(filename , img_rotate_90_flip)
      idx = idx+1
      img_rotate_180 = rotate_image(img_org , 180)
      filename = 'location/Train_sub_images' + '/' + str(idx) + '.jpg'
      cv2.imwrite(filename , img_rotate_180)
      idx = idx+1
      img_rotate_180_flip = cv2.flip(img_rotate_180 , 0)
      filename = 'location/Train_sub_images' + '/' + str(idx) + '.jpg'
      cv2.imwrite(filename , img_rotate_180_flip)
      idx = idx+1
      img_rotate_270 = rotate_image(img_org , 270)
      filename = 'location/Train_sub_images' + '/' + str(idx) + '.jpg'
      cv2.imwrite(filename , img_rotate_270)
      idx = idx+1
      img_rotate_270_flip = cv2.flip(img_rotate_270 , 0)
      filename = 'location/Train_sub_images' + '/' + str(idx) + '.jpg'
      cv2.imwrite(filename , img_rotate_270_flip)
      idx = idx+1
      x = x + 57
    else:
      x = 0
      y = y + 57
print(idx)

