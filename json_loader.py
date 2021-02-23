import json
import os

class COCO2017_annotation_json(object):
    def __init__(self, json_path=None):
        if json_path is None:
            json_path = "mobisAnnotation.json"
        self.json_path = json_path
        self.myjson = None
        if self.json_path is not None:
            with open(self.json_path) as json_data:
                self.myjson = json.load(json_data)

        print(self.json_path )
        # image file name list to check redundancy
        self.filename_list = []
        for content in self.myjson["images"]:
            self.filename_list.append(content["file_name"])

        self.info_index = 0
        self.info_description = ""
        self.info_url=""
        self.info_version=""
        self.info_year=0
        self.info_contributor=""
        self.info_date_created=""
        
        self.licenses_index = 0
        self.licenses_url=""
        self.licenses_id=0
        self.licenses_name=""
        
        self.images_index = 0
        self.images_license=0
        self.images_file_name = ""
        self.images_coco_url=""
        self.images_height=0
        self.images_width=0
        self.images_date_captured=""
        self.images_flickr_url=""
        self.images_id=0
        
        self.annotations_index=0
        self.annotations_segmentation=[]
        self.annotations_num_keypoints=0
        self.annotations_area=0.0
        self.annotations_iscrowd=0
        self.annotations_keypoints=[]
        self.annotations_image_id=0
        self.annotations_bbox=[]
        self.annotations_category_id=0
        self.annotations_id=0
        
        self.categories_index=0
        self.categories_supercategory=""
        self.categories_id=0
        self.categories_name=""
        self.categories_keypoints=[]
        self.categories_skeleton=[]
    
    def get_INFO(self):# key1, one big dict, containing 6 items
        return self.myjson["info"]
    
    def get_LICENSES(self):# key2, one big list, containing 5 dicts
        return self.myjson["licenses"]
    
    #####################################################################
    def get_IMAGES(self): # key3, one big list, containing bunch of dicts
        return self.myjson["images"]
    
    def set_IMAGES(self, images_dict):
        new_comer = images_dict["file_name"] # ex : "043image_display_2687.png"
        other_comer_name = str(''.join(filter(str.isdigit, new_comer))) + (os.path.splitext(new_comer)[1]) # ex : "0432687.png"
        #or str(''.join(filter(str.isdigit, now_file_name))) == (os.path.splitext(str(content["file_name"]) )[0]):
        #filename_list : 044image_display_10005.png
        if (new_comer in self.filename_list) or (other_comer_name in self.filename_list):

            i = 0
            for content in self.myjson["images"]:
                #print("content, new_comer : ", content["file_name"] , new_comer)
                if str(content["file_name"]) == str(new_comer) or str(content["file_name"]) == str(other_comer_name):
                    print("correct , update the content....")
                    ((self.myjson["images"])[i]).update(images_dict)
                    #self.filename_list.append(new_comer)
                    #self.myjson["images"].append(images_dict)
                    break

                else :
                    i = i+1

        else:
            print("that file name is not in filename_list<<<<<<<<<<<<<<<",new_comer, other_comer_name)
            self.filename_list.append(new_comer)
            self.myjson["images"].append(images_dict)
    
    ######################################################################
    def get_ANNOTATIONS(self):# key4, one big list, containing bunch of dicts
        return self.myjson["annotations"]
    
    def set_ANNOTATIONS(self, annotations_dict):
        # 중복제거 기능은 set_IMAGES 함수가 함께 해줌

        #new_person_id = annotations_dict["person_id"]
        #new_image_id = annotations_dict["image_id"]
        new_id =  annotations_dict["id"]
        print("len : ", len(self.myjson["annotations"])-1)

        val = False
        for person_idx in range(len(self.myjson["annotations"])-2):
            #print("person_idx : ", person_idx)
            #print("now_person : ", (self.myjson["annotations"])[person_idx])
            try:
                #if ((self.myjson["annotations"])[person_idx])["person_id"] == new_person_id and ((self.myjson["annotations"])[person_idx])["image_id"] == new_image_id :
                if ((self.myjson["annotations"])[person_idx])["id"] == new_id:
                    print("========= update the annotation id: ", new_id)
                    ((self.myjson["annotations"])[person_idx]).update(annotations_dict)
                    val = True
            except:
                pass

        if val == False:
            self.myjson["annotations"].append(annotations_dict)
        
    def writer(self):
        pass
    
    def saver(self, out_path):
        print(">>>>>>>>>>>>>>outpath : ", out_path)
        with open(out_path, 'w', encoding="utf-8") as json_data:
            json.dump(self.myjson, json_data, ensure_ascii=False, indent='\t')

if __name__ == "__main__":    
    myclass = COCO2017_annotation_json("annotation.json")
    #print(myclass.get_ANNOTATION())