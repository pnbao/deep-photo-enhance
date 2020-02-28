from flask import Flask, request, redirect, url_for, flash, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import LPGAN
from LPGAN import TF, TF_HDR

UPLOAD_FOLDER = 'data/input/'
PROCESSED_FOLDER = 'data/output/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
	return 'Hello, World!'

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/', methods=['GET', 'POST'])
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
			return redirect(url_for('process', photo=filename))
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form method=post enctype=multipart/form-data>
		<input type=file name=file>
		<input type=submit value=Upload>
	</form>
	'''

@app.route('/process/<photo>', methods=['GET', 'POST'])
def process(photo):
	print("Processing enhance image " + photo)
	file_out_no_ext = os.path.splitext(photo)[0]+"-enhanced"
	print("before get input")
	file_name = TF.getInputPhoto(photo)
	print("after get input, before process")
	TF.processImg(file_name , file_out_no_ext)
	#enhanced_img = file_out_no_ext + '.png'
	enhanced_img = send_from_directory(app.config['PROCESSED_FOLDER'], file_out_no_ext + '.png')

	photo_HDR = os.path.splitext(photo)[0]+".png"
	print("Processing HDR image " + photo_HDR)
	file_out_no_ext_HDR = os.path.splitext(photo_HDR)[0]+"-HDR"
	file_name_HDR = TF_HDR.getInputPhoto(photo_HDR)
	TF_HDR.processImg(file_name_HDR , file_out_no_ext_HDR)
	#hdr_img  = file_out_no_ext_HDR + '.png'
	hdr_img = send_from_directory(app.config['PROCESSED_FOLDER'], file_out_no_ext_HDR + '.png')
	return enhanced_img, hdr_img

if __name__ == '__main__':
	app.run(debug=True)



