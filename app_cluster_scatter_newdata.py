# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from sklearn.cluster import KMeans
import math
from dash.dependencies import Input, Output, State
import datetime

df = pd.read_csv("JData_0926.csv",index_col=0, low_memory=False)
df['ClosingDate'] = pd.to_datetime(df['ClosingDate'])
df['ListedDate'] = pd.to_datetime(df['ListedDate'])
df['Type']=df['Type'].replace('CONDP','COOP')
mapbox_access_token = "pk.eyJ1IjoibG5pa2UxIiwiYSI6ImNqOHEwb3dwajBuengycm8wcnR2bjM0NzIifQ.0TzrVKzKLDnaFxwzddgt6g"


def gen_color(line, c_range, c_min):
    c = (line-c_min)/c_range*510
    c2 = 255 - (c-255)*int(c>255)
    c1 = min(c,255)
    return 'rgba('+str(int(c1))+','+str(int(c2))+',20,1)'


def intWithCommas(x):
    if x < 0:
        return '-' + intWithCommas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "$%d%s" % (x, result)


def intWithCommas_n(x):
    if x < 0:
        return '-' + intWithCommas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "%d%s" % (x, result)


def mult_selections(selections_list, column, df):
    if type(selections_list)==str:
        selections_list = [selections_list]
    index_label = []
    for i in selections_list:
        index_label.append(list(df[column]==i))
    index_array = np.array(index_label)
    return index_array.sum(axis=0)>0


def gen_marker(line):
    s = ''
    for i in range(len(line.types)):
        s+= str(line.types[i])+':'+str(line.nums[i])+':'+str(line.other[i])+':'+str(line.counter[i])+'\n'
    return s


def find_nei(nei, df_view):
    for i in range(len(df_view['types'])):
        if nei in df_view['types'][i]:
            return i


def get_meter_chart(a,title):
    if np.isinf(a):
        txt = "inf"
        string = get_shape(30)
    else:
        a = min(30,a)
        string = get_shape(a)
        txt = str(a)[:4]
    base_chart = {
        "values": [36, 54, 54, 54, 54, 54, 54],
        "labels": ["-", "0-Sell", "6", "12", "18", "24", "30-Buy"],
        "domain": {"x": [0, 1]},
        "marker": {
            "colors": [
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)'
            ],
            "line": {
                "width": 1
            }
        },
        "name": "Gauge",
        "hole": .7,
        "type": "pie",
        "direction": "clockwise",
        "rotation": 197,
        "showlegend": False,
        "hoverinfo": "none",
        "textinfo": "label",
        "textposition": "outside"
    }

    meter_chart = {
        "values": [20, 12, 12, 12, 12, 12],
        "labels": [title, "level 1", "level 2", "level 3", "level 4", "level 5"],
        "marker": {
            'colors': [
                'rgb(255, 255, 255)',
                'rgb(232,226,202)',
                'rgb(226,210,172)',
                'rgb(223,189,139)',
                'rgb(223,162,103)',
                'rgb(226,126,64)'
            ]
        },
        "domain": {"x": [0,1]},
        "name": "Gauge",
        "hole": .8,
        "type": "pie",
        "direction": "clockwise",
        "rotation": 135,
        "showlegend": False,
        "textinfo": "label",
        "textposition": "inside",
        "hoverinfo": "none"
    }
    layout = {
        'xaxis': {
            'showticklabels': False,
            'autotick': False,
            'showgrid': False,
            'zeroline': False,
        },
        'yaxis': {
            'showticklabels': False,
            'autotick': False,
            'showgrid': False,
            'zeroline': False,
        },
        'shapes': [
            {
                'type': 'path',
                'path': string,
                'fillcolor': 'rgba(0, 0, 0, 1)',
                'line': {
                    'width': 0.9
                },
                'xref': 'paper',
                'yref': 'paper'
            }
        ],
        'annotations': [
            {
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.45,
                'text': txt,
                'showarrow': False
            }
        ]
    }

    # we don't want the boundary now
    base_chart['marker']['line']['width'] = 0

    fig = {"data": [base_chart, meter_chart],
           "layout": layout}
    return fig


def axis_change(x,y):
    x = x*41/50
    x = x+0.5
    y = y+0.5
    return [str(x),str(y)]


def polar_axis(theta, r):
    theta_pi = theta/360*2*np.pi
    x = np.cos(theta_pi)*r
    y = np.sin(theta_pi)*r
    return axis_change(x,y)


def get_shape(a):
    theta_1 = 225 - a/30*270
    theta_left = 315 - a/30*270
    theta_right = 135 - a/30*270
    points = polar_axis(theta_left,0.01)+polar_axis(theta_1, 0.3)+polar_axis(theta_right, 0.01)
    string = 'M '+' '.join(points[:2])+' L '+' '.join(points[2:4])+' L '+' '.join(points[4:6])+' Z'
    return string

app = dash.Dash(__name__)
server = app.server
server.secret_key = os.environ.get('SECRET_KEY', 'my-secret-key')

app.config.supress_callback_exceptions = True

colors = {
    'background': '#ffffff',
    'text': '#7FDBFF'
}

###webpage layout####

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

layout_page_1 = html.Div([
    html.H1('title'),
    dcc.Graph(id='chart-1',animate = True),
    html.Br(),
    dcc.Link('Home', href='/')
])

index_page = html.Div(style={'backgroundColor': colors['background'],
                             'font-family': 'Arial',
                             'textAlign': 'left'},
                      children=[
    html.H1(
        children='Data Analysis',
        style={
            'textAlign': 'center',
            'borderBottom': 'thin lightgrey solid'
        }
    ),

    html.Label('Neighborhoods'),
    dcc.Dropdown(
        id = 'neighborhood-sel',
        options=[{'label': i, 'value': i} for i in df['Neighborhood'].unique()],
        value=['Upper East Side'],
        multi=True
    ),

    html.Div([

    html.Table(id='table'),

    ]),


    dcc.Graph(
            id='avg-dollar-sqft',
            animate = True),

    html.Br(),
    dcc.Link('Chart_1', href='/page-1'),

    html.Div([
    html.P('Beds'),
    dcc.Checklist(
        id = 'beds-checklist',
        options=[
            {'label': '0', 'value': 0},
            {'label': '1', 'value': 1},
            {'label': '2', 'value': 2},
            {'label': '3', 'value': 3},
            {'label': '4', 'value': 4},
            {'label': '5+', 'value': 5}
        ],
        values=[1,2]
    )]),

    html.Div([
    html.P('Types'),
    dcc.Checklist(
        id = 'types-checklist',
        options=[
            {'label': 'Condo', 'value': 'CONDO'},
            {'label': 'Coop', 'value': 'COOP'},
            {'label': 'SingleFamily', 'value':'SFAM'},
            {'label': 'MultiFamily', 'value':'MULTI'},
        ],
        values=['CONDO','COOP']
    )]),

    html.Hr(style={'opacity': 0}),
    html.Hr(style={'opacity': 0}),
    html.P('Closing Price'),
    dcc.RangeSlider(
            id = 'closingprice-slider',
            marks={i: '{}'.format(intWithCommas(i)) for i in range(0, 10000000, 500000)},
            min=0,
            max= 10000000,
            value=[50000, 10000000],
            # step=None,
            #style={'width': 100, 'height': 10}
    ),

    html.Hr(style={'opacity': 0}),
    html.Hr(style={'opacity': 0}),
    html.P('Square Feet'),
    dcc.RangeSlider(
                id = 'sqft-slider',
                min=0,
                max=8000,
                value=[0, 3000],#max(df['SqFt'])
                marks={i: '{}SqFt'.format(intWithCommas_n(i)) for i in range(0, 8000, 500)},
                # step=None
                ),
    html.Hr(style={'opacity': 0}),
    html.Hr(style={'opacity': 0}),
    html.P('Dollar/SqFt'),
    dcc.RangeSlider(
                id = 'dollarsqft-slider',
                min=0,
                max=8000,
                value=[0, 3000],#max(df['SqFt'])
                marks={i: '{}'.format(intWithCommas(i)) for i in range(0, 8000, 500)},
                # step=None
                ),
    html.Hr(style={'opacity': 0}),
    html.Hr(style={'opacity': 0}),
    html.P('Output'),
    dcc.RadioItems(
        id = 'toggle',
        options=[
            {'label': 'AverageDollarSqFt', 'value': 1},
            {'label': 'ClosingPrice', 'value': 2}
        ],
        value=1
    ),
    html.Div([dcc.Graph(id='cluster-graph')],
             style={'width': '64%', 'display': 'inline-block','vertical-align':'top', 'padding': '0 20'}),
    html.Div([
        dcc.Markdown("""
                **Similar Neighborhoods**
            """.replace('   ', '')),
        html.Div(id='hover-data')],
        style={'display': 'inline-block','width': '34%','vertical-align':'top'}),

    html.Button('Click Me', id='button-1'),

    html.Div([dcc.Graph(id='map-graph')]),
    html.Div([dcc.Graph(id='meter-chart-1')],
             style={'width': '34%', 'display': 'inline-block',
                    'vertical-align':'top', 'padding': '0 20'}),

    html.Div([dcc.Graph(id='meter-chart-2')],
           style={'width': '34%', 'display': 'inline-block',
                  'vertical-align': 'top', 'padding': '0 20'})

    ])



@app.callback(
    dash.dependencies.Output('meter-chart-1', 'figure'),
    inputs=[Input('sqft-slider', 'value'),
     Input('neighborhood-sel', 'value'),
     Input('closingprice-slider', 'value'),
     Input('beds-checklist', 'values'),
     Input('dollarsqft-slider','value'),
     Input('types-checklist','values'),
     Input('avg-dollar-sqft', 'clickData')])
def absortion_ratio_ave(selected_sqft, selected_neighborhood,
                    selected_cp, selected_beds, selected_ds,
                    selected_types, hoverData):
    beds_labels = mult_selections(selected_beds, 'Beds', df)
    closing_price_labels = (df['ClosingPrice'] < selected_cp[1]) & (df['ClosingPrice'] > selected_cp[0])
    sqft_labels = (df['SqFt'] < selected_sqft[1]) & (df['SqFt'] > selected_sqft[0])
    dollarsqft_label = (df['DollarSqFt'] < selected_ds[1]) & (df['DollarSqFt'] > selected_ds[0])
    types_labels = mult_selections(np.array(selected_types), 'Type', df)
    neighborhood_labels = mult_selections(selected_neighborhood, 'Neighborhood', df)
    labels_1 = beds_labels & closing_price_labels & sqft_labels \
               & dollarsqft_label & neighborhood_labels & types_labels

    df1 = df.loc[labels_1]

    date = hoverData['points'][0]['x']

    date_y = pd.to_datetime(date)
    a_sum = 0
    for i in range(3):
        date_y -= datetime.timedelta(365)
        start = date_y - pd.tseries.offsets.MonthEnd(3)
        end = date_y + pd.tseries.offsets.MonthEnd()

        sales_labels = df1['ClosingDate'].between(start, end)
        inventories_labels = (df1['ClosingDate'] > start) & (df1['ListedDate'] < end)
        a_sum +=  sum(inventories_labels)/sum(sales_labels)
        print(a_sum / 3)

    return get_meter_chart(a_sum / 3, 'Historical')


@app.callback(
    dash.dependencies.Output('meter-chart-2', 'figure'),
    inputs=[Input('sqft-slider', 'value'),
     Input('neighborhood-sel', 'value'),
     Input('closingprice-slider', 'value'),
     Input('beds-checklist', 'values'),
     Input('dollarsqft-slider','value'),
     Input('types-checklist','values'),
     Input('avg-dollar-sqft', 'clickData')])
def absortion_ratio_current(selected_sqft, selected_neighborhood,
                    selected_cp, selected_beds, selected_ds,
                    selected_types, hoverData):
    beds_labels = mult_selections(selected_beds, 'Beds', df)
    closing_price_labels = (df['ClosingPrice'] < selected_cp[1]) & (df['ClosingPrice'] > selected_cp[0])
    sqft_labels = (df['SqFt'] < selected_sqft[1]) & (df['SqFt'] > selected_sqft[0])
    dollarsqft_label = (df['DollarSqFt'] < selected_ds[1]) & (df['DollarSqFt'] > selected_ds[0])
    types_labels = mult_selections(np.array(selected_types), 'Type', df)
    neighborhood_labels = mult_selections(selected_neighborhood, 'Neighborhood', df)
    labels_1 = beds_labels & closing_price_labels & sqft_labels \
               & dollarsqft_label & neighborhood_labels & types_labels

    df1 = df.loc[labels_1]

    date = hoverData['points'][0]['x']
    date_y = pd.to_datetime("2017-09")
    start = date_y - pd.tseries.offsets.MonthEnd(3)
    end = date_y + pd.tseries.offsets.MonthEnd()

    sales_labels = df1['ClosingDate'].between(start, end)
    inventories_labels = (df1['ClosingDate'] > start)
    a = sum(inventories_labels)/sum(sales_labels)
    print(a)
    return get_meter_chart(a, 'Current')


@app.callback(
    dash.dependencies.Output('map-graph', 'figure'),
    [Input('button-1','n_clicks')],
    state=[State('sqft-slider', 'value'),
            State('avg-dollar-sqft', 'clickData'),
            State('closingprice-slider', 'value'),
            State('beds-checklist', 'values'),
            State('dollarsqft-slider','value'),
            State('types-checklist','values'),
            State('toggle', 'value')
    ])
def update_map(n_clicks,selected_sqft, hoverData, selected_cp,
               selected_beds, selected_ds, selected_types, toggle):

    column = {2: 'ClosingPrice', 1: 'DollarSqFt'}[toggle]

    date = hoverData['points'][0]['x']
    date_y = pd.to_datetime(date)
    start = date_y - datetime.timedelta(365)
    end = date_y + datetime.timedelta(35)

    year_labels = df['ClosingDate'].between(start, end)

    if selected_cp[1] > 9500000:
        selected_cp[1] = df['ClosingPrice'].max()+1
    if selected_ds[1] > 7500:
        selected_ds[1] = df['DollarSqFt'].max()+1
    if selected_sqft[1] > 7500:
        selected_sqft[1] = df['SqFt'].max()

    beds_labels = mult_selections(selected_beds, 'Beds', df)
    closing_price_labels = (df['ClosingPrice'] < selected_cp[1]) & (df['ClosingPrice'] > selected_cp[0])
    sqft_labels = (df['SqFt'] < selected_sqft[1]) & (df['SqFt'] > selected_sqft[0])
    dollarsqft_label = (df['DollarSqFt'] < selected_ds[1]) & (df['DollarSqFt'] > selected_ds[0])
    types_labels = mult_selections(np.array(selected_types), 'Type', df)


    labels_1 = beds_labels & closing_price_labels & sqft_labels & dollarsqft_label & types_labels &year_labels
    df_output = df.loc[labels_1][['Lat','Lng','SellersAgent1',column]]

    c_min = df_output[column].min()
    c_range = df_output[column].max() - c_min

    df_output['color'] = df_output[column].apply(gen_color, args=[c_range,c_min])


    data = go.Data([
        go.Scattermapbox(
            lat=df_output['Lat'],
            lon=df_output['Lng'],
            mode='markers',
            marker=go.Marker(
                size=7,
                opacity=0.6,
                color=df_output['color']
            ),
            text=df_output[column].apply(intWithCommas)
        )
    ])
    layout = go.Layout(
        autosize=False,
        width=1300,
        height=800,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=40.717086,
                lon=-73.991040
            ),
            pitch=0,
            zoom=10,
            style='dark'
        ),
    )


    return dict(data=data, layout=layout)

@app.callback(
    dash.dependencies.Output('avg-dollar-sqft', 'figure'),
    inputs=[Input('sqft-slider', 'value'),
     Input('neighborhood-sel', 'value'),
     Input('closingprice-slider', 'value'),
     Input('beds-checklist', 'values'),
     Input('toggle', 'value'),
     Input('dollarsqft-slider','value'),
     Input('types-checklist','values')])
def update_figure(selected_sqft, selected_neighborhood, selected_cp, selected_beds, selected_tog, selected_ds,selected_types):
    if selected_cp[1] > 9500000:
        selected_cp[1] = df['ClosingPrice'].max()+1
    if selected_ds[1] > 7500:
        selected_ds[1] = df['DollarSqFt'].max()+1
    if selected_sqft[1] > 7500:
        selected_sqft[1] = df['SqFt'].max()

    print(selected_beds)
    print('beds')

    print(selected_sqft)
    print('sqft')

    print(selected_neighborhood)
    print('neighbor')

    print(selected_cp)
    print('cp')

    print(selected_tog)
    print('tog')

    print(selected_ds)
    print('ds')

    print(selected_types)
    print('types')



    beds_labels = mult_selections(selected_beds, 'Beds', df)
    closing_price_labels = (df['ClosingPrice'] < selected_cp[1]) & (df['ClosingPrice'] > selected_cp[0])
    sqft_labels = (df['SqFt'] < selected_sqft[1]) & (df['SqFt'] > selected_sqft[0])
    dollarsqft_label = (df['DollarSqFt'] < selected_ds[1]) & (df['DollarSqFt'] > selected_ds[0])
    types_labels = mult_selections(np.array(selected_types), 'Type', df)

    # generate the df_clusters, for bar plot
    labels_1 = beds_labels & closing_price_labels & sqft_labels & dollarsqft_label & types_labels
    df_output = df.loc[labels_1][['DollarSqFt', 'ClosingPrice', 'ClosingDate', 'Neighborhood']]
    df_output.index = df_output['ClosingDate']
    del df_output['ClosingDate']

    df_list = []
    for i in df_output['Neighborhood'].unique():
        df1 = df_output[df_output['Neighborhood'] == i]
        se_tmp = df1.resample('A').mean()
        se_tmp['Neighborhood'] = i
        #         se_tmp['Num'] = len(df1)
        l = []
        for j in se_tmp.index.year:
            if math.isnan(se_tmp[str(j)]['DollarSqFt'][0]) == False:
                l.append(len(df1[str(j)]))
            else:
                l.append(0)
        se_tmp['Num'] = l
        df_list.append(se_tmp)

    global df_clusters
    df_clusters = pd.concat(df_list)

    # generate data for plot line
    traces = []
    for i in range(len(selected_neighborhood)):
        neighborhood_labels = mult_selections([selected_neighborhood[i]], 'Neighborhood', df)
        labels = beds_labels & neighborhood_labels & closing_price_labels & sqft_labels & dollarsqft_label & types_labels

        df_output = df.loc[labels][['DollarSqFt', 'ClosingPrice', 'ClosingDate']]
        df_output.index = df_output['ClosingDate']
        del df_output['ClosingDate']
        df_output = df_output.resample('Q').mean()

        if selected_tog == 1:
            y1 = df_output['DollarSqFt'].interpolate()
            y1axis = {'title': 'Avg$/SqFt'}
        else:
            y1 = df_output['ClosingPrice'].interpolate()
            y1axis = {'title': 'ClosingPrice'}

        strtdate = {'1': '-03-31', '2': '-06-30', '3': '-09-30', '4': '-12-31'}

        traces.append(go.Scatter(
            customdata=[selected_neighborhood[i] for j in range(len(y1))],
            x=[str(i) for i in list(y1.index.year.astype(str) + [strtdate[j] for j in y1.index.quarter.astype(str)])],
            y=y1.values,
            text=[k for k in list(y1.index.year.astype(str) + [strtdate[j] for j in y1.index.quarter.astype(str)])],
            mode='lines',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=selected_neighborhood[i]
        ))
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'Time'},
            yaxis= y1axis)

    }

def comma(x):
    if len(x)!=0:
        return x[:-2]
    return x

def get_average(df_clusters, date, column, neighbor):
    # choose the year
    date_y = date[:4]
    date_m = date[4:7]
    df1 = df_clusters.loc[date_y].set_index('Neighborhood').dropna()

    # choose the column
    other = "DollarSqFt"
    if column == "DollarSqFt":
        other = "ClosingPrice"

    n_clusters = max(int(len(df1) / 10), 10)
    if int(len(df1)) < 10:
        n_clusters = int(len(df1))

    txt = dict([(i, []) for i in range(n_clusters)])
    val1 = dict([(i, []) for i in range(n_clusters)])
    val2 = dict([(i, []) for i in range(n_clusters)])
    val3 = dict([(i, []) for i in range(n_clusters)])

    s = KMeans(n_clusters=n_clusters, random_state=0).fit(df1[column].values.reshape(len(df1), 1))
    for i in range(len(df1)):
        txt[s.labels_[i]].append(str(df1.index[i]))
        val1[s.labels_[i]].append(df1[column].values[i])
        val2[s.labels_[i]].append(df1[other].values[i])
        val3[s.labels_[i]].append(df1['Num'].values[i])

    global df_view

    df_view = pd.DataFrame({'types': txt, 'nums': val1, 'other': val2, 'counter': val3})
    df_view['means'] = df_view['nums'].apply(np.mean)
    df_view['markers'] = df_view.apply(gen_marker, axis=1)
    df_view.sort_values(by='means', inplace=True)
    df_view.index = range(len(df_view))

    color_label = [0 * i for i in range(n_clusters)]
    text = ['' for i in range(n_clusters)]
    for nei in neighbor:
        color_label += df_view['types'].apply(lambda x: int(nei in x))
        text[find_nei(nei, df_view)] += nei + ', '
    colors = dict([(0, 'rgba(204,204,204,1)')] + [(i, 'rgba(222,45,38,0.8)') for i in range(1, len(neighbor)+2)])

    # print('colors')
    # print(colors)
    #
    # print('neighbor')
    # print(neighbor)
    #
    # print('color_label')
    # print(color_label)

    text = list(map(comma,text))
    data = [go.Bar(
        x=[str(x) for x in range(n_clusters)],
        y=df_view['means'],
        text=text,
        marker=dict(
            color=[colors[i] for i in color_label],
        ))]

    layout = go.Layout(

        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Ave' + column
        ),
        bargap=0,
        bargroupgap=0.1
    )

    return{
    'data':data ,

    'layout' : layout}


@app.callback(
    dash.dependencies.Output('cluster-graph', 'figure'),
    [dash.dependencies.Input('avg-dollar-sqft', 'clickData'),
    dash.dependencies.Input('toggle', 'value')])
def cluster_figure(hoverData, toggle):
    neighbor = [hoverData['points'][i]['customdata'] for i in range(len(hoverData['points']))]
    date = hoverData['points'][0]['x']
    column = {2:'ClosingPrice', 1:'DollarSqFt'}[toggle]
    return get_average(df_clusters, date, column, neighbor)


def df_ftxt(txt, column):
    # choose the column
    other = "DollarSqFt"
    if column == "DollarSqFt":
        other = "ClosingPrice"


    dict_list = []
    for t in txt.split('\n'):
        if t != '':
            tmp = t.split(':')
            dict_list.append({'Neighbor':tmp[0], column:tmp[1], other:tmp[2], 'Num':tmp[3]})
    # print(dict_list)
    # print(column)
    dataframe = pd.DataFrame.from_records(dict_list)
    dataframe['DollarSqFt']=pd.to_numeric(dataframe['DollarSqFt']).astype(int)
    dataframe['ClosingPrice']=pd.to_numeric(dataframe['ClosingPrice']).astype(int)
    dataframe['SqFt'] = (dataframe['ClosingPrice'] / dataframe['DollarSqFt']).astype(int)

    # print(dataframe)

    dataframe = dataframe.sort_values(by=column, ascending=False)

    dataframe['DollarSqFt'] = dataframe['DollarSqFt'].apply(intWithCommas)
    dataframe['ClosingPrice']=dataframe['ClosingPrice'].apply(intWithCommas)
    dataframe['SqFt']=dataframe['SqFt'].apply(intWithCommas_n)
    dataframe['Num'] = pd.to_numeric(dataframe['Num']).astype(int)

    dataframe = dataframe[['Num','Neighbor','ClosingPrice','DollarSqFt','SqFt']]

    # print(dataframe)
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))]
    )

@app.callback(
    dash.dependencies.Output('hover-data', 'children'),
    [dash.dependencies.Input('cluster-graph', 'hoverData'),
     dash.dependencies.Input('toggle', 'value')])
def get_cluster_info(hoverData, toggle):
    i = hoverData['points'][0]['x']
    txt = df_view.loc[i]['markers']
    column = {2: 'ClosingPrice', 1: 'DollarSqFt'}[toggle]
    print(i)
    return df_ftxt(txt, column)




# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return layout_page_1
    else:
        return index_page




if __name__ == '__main__':
    app.run_server()

