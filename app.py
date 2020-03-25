from flask import Flask, request, redirect, url_for, flash, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'static/data/input/'
PROCESSED_FOLDER = 'static/data/output/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_url_path='/static')

CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
	# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('process', photo=os.path.basename(filename)))
	return render_template('index.html')

@app.route('/process/<photo>', methods=['GET', 'POST'])
def process(photo):
    from LPGAN import TF, FUNCTION
    file_name = TF.getInputPhoto(photo)
    print(FUNCTION.current_time() + "Processing image " + file_name)
    file_out_no_ext = os.path.splitext(photo)[0]+"-enhanced"
    enhanced_img_name = TF.processImg(file_name , file_out_no_ext)
    enhanced_img = os.path.join(PROCESSED_FOLDER, enhanced_img_name)

    from LPGAN import TF_HDR
    print(FUNCTION.current_time() + "Processing HDR image " + file_name)
    file_out_no_ext_HDR = os.path.splitext(file_name)[0]+"-HDR"
    hdr_img_name = TF_HDR.processImg(file_name , file_out_no_ext_HDR)
    hdr_img  = os.path.join(PROCESSED_FOLDER, hdr_img_name)

    print(FUNCTION.current_time(), enhanced_img, hdr_img)

    original_img = os.path.join(UPLOAD_FOLDER, file_name)
    return render_template('result.html', original_img="/"+original_img, enhanced_img="/"+enhanced_img, hdr_img="/"+hdr_img)

if __name__ == '__main__':
	app.run(debug=True)



