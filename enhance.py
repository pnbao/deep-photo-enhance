from LPGAN import TF, TF_HDR, FUNCTION
import os

UPLOAD_FOLDER = 'static/data/input'
PROCESSED_FOLDER = 'static/data/output'

photo = 'news.png'
file_name = TF.getInputPhoto(photo)

print(FUNCTION.current_time() + "Processing image " + file_name)
file_out_no_ext = os.path.splitext(photo)[0]+"-enhanced"
enhanced_img_name = TF.processImg(file_name , file_out_no_ext)
enhanced_img = os.path.join(PROCESSED_FOLDER, enhanced_img_name)

print(FUNCTION.current_time() + "Processing HDR image " + file_name)
file_out_no_ext_HDR = os.path.splitext(file_name)[0]+"-HDR"
hdr_img_name = TF_HDR.processImg(file_name , file_out_no_ext_HDR)
hdr_img  = os.path.join(PROCESSED_FOLDER, hdr_img_name)

print(FUNCTION.current_time(), enhanced_img, hdr_img)
