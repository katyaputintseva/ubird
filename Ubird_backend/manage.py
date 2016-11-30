from flask import Flask, render_template, request, make_response
from werkzeug import secure_filename
import os
import data_vis_function
import Egor

app = Flask(__name__)

# HOME page with "Choose File" & "Submit" buttons
@app.route('/')
def upload_page():
   return render_template('home.html')

# RESULTS page with read/write file, static matplotlib histogram, and bokeh plots
@app.route('/results', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save("uploads//" + secure_filename(f.filename))
      ML = Egor.MachineLearningPipeline("uploads//" + secure_filename(f.filename))
      mysb, mydb = data_vis_function.result("uploads//" + secure_filename(f.filename))
      return render_template('results.html', content=ML, mybok=mysb,mydiv=mydb)

if __name__ == '__main__':
   app.run(debug = True)
