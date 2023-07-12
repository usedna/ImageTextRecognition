import os

import numpy as np
import cv2.cv2 as cv2
from OCR.utils import predict_image
import tensorflow as tf

############################
width = 640
height = 480
threshold = 0.65
###########################


version_test = ((50, 64), (50, 128), (50, 256),
                (100, 128), (100, 256),
                (125, 64), (125, 128), (125, 256), (125, 1024),
                (200, 128), (200, 256), (200, 512),(200, 1024),
                (500, 1024))

for i, ep_bs in enumerate(version_test[-1:]):
       epoch, batch_size = ep_bs
       ocr_model = tf.keras.models.load_model(f'path_to_model')
       fail = 0
       absent = 0
       correct = 0
       wrong = 0
       failures = {}
       for f in os.listdir('../Img'):
              val = int(f.split('g')[1][:3]) - 1


              if val not in failures.keys():
                     failures.update({val: [0, []]})

              img_original = cv2.imread(f"../Img/{f}")
              img = np.asarray(img_original)
              letter, class_indexes = predict_image(img, ocr_model)
              # print(f, letter)
              # cv2.putText(img_original, letter[0], (50, 50),
              #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

              if class_indexes is None:
                     fail += 1
                     absent += 1
                     failures[val][0] += 1
                     failures[val][1].append((f, 'absent'))
              elif val != class_indexes:
                     fail += 1
                     wrong += 1
                     failures[val][0] += 1
                     failures[val][1].append((f, 'wrong'))
              else:
                     correct += 1
       with open('test_result.txt', 'a') as ft:
              print(f"-------------------TEST{i}-----------------------", file=ft)
              print(f"v{epoch}bs{batch_size}", file=ft)
              print("Accuracy:", fail / (correct + fail), file=ft)
              print("False negative(%):", absent / fail, file=ft)
              print("False positivie(%):", wrong / fail, file=ft)
              print("Recall:", correct / (correct + absent), file=ft)
              print("Precision:", correct / (correct + wrong), file=ft)
              print("Toatal fails:", failures, file=ft)
              print(file=ft)
       print("Done test", i)
