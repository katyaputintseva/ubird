from os.path import dirname, join

import pandas as pd

from bokeh.layouts import column, widgetbox, WidgetBox
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Button, DataTable, TableColumn, NumberFormatter
from bokeh.models.layouts import HBox
from bokeh.io import curdoc
import rf_functions
from bokeh.embed import components

def get_table(df):

    results = rf_functions.MachineLearningPipeline(df)

    res_df = pd.DataFrame(results)
    res_df.columns = [['mutation','confidence','ocurrence']]

    source = ColumnDataSource(res_df)


    button = Button(label="Download", button_type="success")
    button.callback = CustomJS(args=dict(source=source),
                               code=open(join(dirname(__file__), "static/download.js")).read())

    columns = [
        TableColumn(field="mutation", title="Mutation name"),
        TableColumn(field="confidence", title="Predictor confidence", formatter=NumberFormatter(format="0.00000")),
        TableColumn(field="ocurrence", title="Ocurrence of mutations")
    ]

    data_table = DataTable(source=source, columns=columns, width=600, fit_columns=True)

    controls = widgetbox(button)
    table = widgetbox(data_table)

    scr, div = components(column(controls, table))
    return str(scr), str(div)



