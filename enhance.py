import LPGAN
from LPGAN import TF, TF_HDR
import os

UPLOAD_FOLDER = 'data/input'
PROCESSED_FOLDER = 'data/output'

#test_dir = "data/input/"
#test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]


#for photo in test_photos:
photo = 'small.png'
print("Processing image " + photo)
file_out_no_ext = os.path.splitext(photo)[0]+"-enhanced"
file_name = TF.getInputPhoto(photo)
TF.processImg(file_name , file_out_no_ext)
enhanced_img = os.path.join(PROCESSED_FOLDER, file_out_no_ext + '.png')
	
photo_HDR = photo
print("Processing HDR image " + photo_HDR)
file_out_no_ext_HDR = os.path.splitext(photo_HDR)[0]+"-HDR"
file_name_HDR = TF_HDR.getInputPhoto(photo_HDR)
TF_HDR.processImg(file_name_HDR , file_out_no_ext_HDR)
hdr_img  = os.path.join(PROCESSED_FOLDER, file_out_no_ext_HDR + '.png')
print(enhanced_img, hdr_img)
