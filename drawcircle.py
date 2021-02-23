import cv2
import os, sys
import json

dirname = "output/"
filenames = os.listdir(dirname)

myImageList = list()
for filename in filenames:
    if filename.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        full_filename = os.path.join(dirname, filename)
        myImageList.append(full_filename)

json_path = "mobisAnnotation2.json"
json_data = open(json_path)
myjson = json.load(json_data)

# image file name list to check redundancy
filename_list = []
for content in myjson["images"]:
    filename_list.append(content["file_name"])

inputKeypointList = [[0] * 3 for i in range(10)]
'''
for annotationContent in myjson["annotations"]:
    for i in range(10):
        inputKeypointList[i][0] = (annotationContent["keypoints"])[i * 3]
        inputKeypointList[i][1] = (annotationContent["keypoints"])[i * 3 + 1]
        inputKeypointList[i][2] = (annotationContent["keypoints"])[i * 3 + 2]
'''

for annotationContent in myjson["annotations"]:
    print("try circle " + annotationContent["image_id"])

    for i in range(10):
        inputKeypointList[i][0] = (annotationContent["keypoints"])[i * 3]
        inputKeypointList[i][1] = (annotationContent["keypoints"])[i * 3 + 1]
        inputKeypointList[i][2] = (annotationContent["keypoints"])[i * 3 + 2]

    for img_name in myImageList:

        if annotationContent["image_id"] == os.path.splitext(os.path.split(img_name)[1])[0] :

            img = cv2.imread(img_name)
            for i in range(8):
                #if inputKeypointList[i][2] == 1 :
                 if i <5 :
                     cv2.circle(img, (int(inputKeypointList[i][0]), int(inputKeypointList[i][1])), 7,  (0, 0, 255), 1)
                 else:
                     cv2.circle(img, (int(inputKeypointList[i][0]), int(inputKeypointList[i][1])), 10,  (255, 0, 0), 1)

            cv2.imwrite( "output/" + os.path.splitext(os.path.split(img_name)[1])[0] +".png" , img)



