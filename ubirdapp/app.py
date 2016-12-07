from flask import Flask, render_template, request, make_response
from werkzeug import secure_filename
import os
import data_vis_function
import to_table


app = Flask(__name__)
app._static_folder = 'static/'

# HOME page with "Choose File" & "Submit" buttons
@app.route('/')
def upload_page():
    return render_template('home.html')



#   N E W : separation of plots for better frontend manipulation
@app.route('/results', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("uploads//" + secure_filename(f.filename))
        df = data_vis_function.read_file("uploads//" + secure_filename(f.filename))

        # Random forest Results:
        # PUT IN TABLE FOR CLIENT VISUALIZATION 
        ## pos_mutaions and neg_mutations are list of tuples. Each tuple contains 4 values:
        #   - mutation name
        #   - prediction confidence of classifier
        #   - occurence of mutation in the dataset
        #   - importance of mutation as feature for the classifier

        # Definition of script and div for each plot function {--str(script),str(div)--}
        bardist_scr, bardist_div = data_vis_function.result_barcode_dist(df)
        mutdist_scr, mutdist_div = data_vis_function.result_mut_dist(df)
        aminos_scr, aminos_div = data_vis_function.result_amino_switch(df)
        uniqmut_scr, uniqmut_div = data_vis_function.result_uniq_mut(df)
        violin_scr, violin_div = data_vis_function.result_violin(df)
        table_scr, table_div = to_table.get_table(df)
        # mysb, mydb = data_vis_function.result(df)
        return render_template('results3.html',
                               bardist_scr=bardist_scr, mutdist_scr=mutdist_scr, aminos_scr=aminos_scr,
                               uniqmut_scr=uniqmut_scr, violin_scr=violin_scr, table_scr=table_scr,
                               bardist_div=bardist_div, mutdist_div=mutdist_div,
                               aminos_div=aminos_div, uniqmut_div=uniqmut_div, violin_div=violin_div,
                               table_div=table_div)# results_table=ML,



# # RESULTS page with read/write file, static matplotlib histogram, and bokeh plots
# @app.route('/results', methods = ['GET', 'POST'])
# def upload_file():
# 	if request.method == 'POST':

#         f = request.files['file']
#         f.save("uploads//" + secure_filename(f.filename))
#         df = data_vis_function.read_file("uploads//" + secure_filename(f.filename))
#         ML = Egor.MachineLearningPipeline(df)
#         mysb, mydb = data_vis_function.result(df)
#         return render_template('results.html', content=ML, mybok=mysb,mydiv=mydb)

if __name__ == '__main__':
    app.run(debug=True)
