import json
from pprint import pprint
from collections import OrderedDict

class Editor():

    def __init__(self):
        super(Editor, self).__init__()
        #self.setupUi(self)

        self.file_data = OrderedDict()

        self.file_data["info"] = {}
        self.file_data["info"]["description"] = "COCO 2017 Dataset"
        self.file_data["info"]["url"] = "http://cocodataset.org"
        self.file_data["info"]["version"] = "1.0"
        self.file_data["info"]["year"] = 2017
        self.file_data["info"]["contributor"] = "COCO Consortium"
        self.file_data["info"]["date_created"] = "2017/09/01"

        self.file_data["licenses"] = []
        self.file_data["licenses"].append({"url" : "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"})
        self.file_data["licenses"].append({"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2,"name": "Attribution-NonCommercial License"})
        self.file_data["licenses"].append({"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3, "name": "Attribution-NonCommercial-NoDerivs License"})
        self.file_data["licenses"].append({"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License"})
        self.file_data["licenses"].append({"url": "http://www.usa.gov/copyright.shtml", "id": 8, "name": "United States Government Work"})


        self.file_data["images"] = []
        #self.file_data["images"].append({})

        '''
                self.file_data["images"]["license"] = 4
                self.file_data["images"]["file_name"] = ""
                #self.file_data["images"]["coco_url"] = ""
                self.file_data["images"]["height"] = ""
                self.file_data["images"]["width"] = ""
                self.file_data["images"]["date_captured"] = ""
                #self.file_data["images"]["flickr_url"] = ""
        '''

        self.file_data["annotations"] = []
        #self.file_data["annotations"].append({})

        '''
        self.file_data["annotations"]["segmentation"] = ""
        self.file_data["annotations"]["num_keypoints"] = ""
        self.file_data["annotations"]["image_id"] = ""
        self.file_data["annotations"]["bbox"] = ""
        '''

        temp_cat = {}
        temp_cat["supercategory"] = "person"
        temp_cat["id"] = 1
        temp_cat["name"] = "person"
        temp_cat["keypoints"] = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                                 "right_shoulder", "left_elbow", "right_elbow"]
        temp_cat["skeleton"] = [[3, 1], [1, 0], [0, 2], [2, 4], [0, 5], [8, 6], [6, 5], [5, 7], [7, 9]]

        self.file_data["categories"] = []
        self.file_data["categories"].append(temp_cat)
