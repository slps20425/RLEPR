import dash
import dash_html_components as html
import flask 
from flask_caching import Cache
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_table
import dash_table_experiments as dt
import numpy as np



tableWeight = pd.DataFrame()
class Percent(float):
    def __str__(self):
        return '{:.2%}'.format(self)



# preprocessing
def preprocessWeight(datasetDF):
    datasetDF.index = pd.to_datetime(datasetDF.index, format = '%Y-%m-%d').strftime('%Y-%m-%d')

    
    daily_pct_change = datasetDF.Asset.pct_change()
    daily_pct_change.fillna(0, inplace=True)
    cumprod_daily_pct_change = (1 + daily_pct_change).cumprod()
    datasetDF['ROI'] = cumprod_daily_pct_change

    t_change = price_change_df.loc[datasetDF.index.tolist()]  * datasetDF[datasetDF.columns[1:-3]].astype(float) 
    datasetDF['WeeklyROI'] = t_change.sum(axis = 1) #(1+t_change.sum(axis = 1)).cumprod()-1








    weight=datasetDF.iloc[::-1]
    weight = weight.applymap("{0:.2f}".format)
    weight=weight.rename(columns={0: 'Cash'})
    
    weight = weight.replace('0.00', '-')
      
    return weight


def redesignWeight(weight):
    global tableWeight
    defaultWeight =weight.loc[:,[weight.columns[0],weight.columns[1],weight.columns[-1],weight.columns[-2],weight.columns[-3]]]
    defaultWeight.Asset = defaultWeight.Asset.astype(float).map('{:,.2f}'.format) 
    selectedWeight = weight.iloc[:, 2:-3].iloc[:,:-1]
    keys=selectedWeight.apply(lambda row: row[row != '-'].index, axis=1)
    keys =keys.apply(lambda x: x.astype('int32'))
    
    keyy=pd.DataFrame(keys, columns=['keys'])
    newly_add = keyy['keys'].apply(set).diff(-1).fillna('')
    newly_drop = keyy['keys'].apply(set).diff(1).shift(-1).fillna('')

    keys =keys.apply(lambda x: x.astype(str))
    newly_add = newly_add.apply(lambda y: np.nan if len(y)==0 else y).fillna('').values.astype(str)
    newly_drop = newly_drop.apply(lambda y: np.nan if len(y)==0 else y).fillna('').values.astype(str)
    
    values=selectedWeight.apply(lambda row: row[row != '-'].values.astype(float), axis=1)
    values=values.apply(lambda x: (x*100))
    values=values.apply(lambda x: [str(val) for val in x])

    # values= values.apply(lambda x: [format(val, '.2f') for val in x]) # the ans from chatGPT
    # values= values.apply(lambda x: [Percent(val) for val in x]) 
    # values =values.apply(lambda x: x.astype(str))

    keys = keys +':'
    newElement =  keys + values + '%'
    defaultWeight['Portfolio'] = newElement.astype(str).values
    defaultWeight['Add'] = newly_add
    defaultWeight['Drop'] = newly_drop
    defaultWeight['Portfolio']=defaultWeight['Portfolio'].apply(lambda x : x[7:-18])
    defaultWeight.Cash= defaultWeight.Cash.replace('-','0.0').astype(float).map("{:.1%}".format)
    new_col = ['Date \\ ','Add', 'Drop','Portfolio', 'Cash','Asset', 'WeeklyROI', 'ROI']
    defaultWeight.index= pd.to_datetime(weight[weight.columns[0]])

    tableWeight = defaultWeight[new_col]
    return tableWeight

# Load data
processed_data_path='./data/latest-45tic-data-preprocess.pkl'
pricebook = pd.read_pickle('./data/latest_45tic_priceBook.pkl')
df = pricebook
df = df.loc['2019-01-01':]
price_change_df = pd.pivot_table(df,index=df.index,columns=df.Stock_ID,values='close')



RLweight = pd.read_pickle('./data/latest_test_live_update.pkl')
# RLweight = RLweight['allocation_targets']


# Load Jiang's data
jiang_weights = pd.read_pickle('./data/latest_jiang_weights_1649492778.pkl')
# inter_dateRange= np.intersect1d(jiang_weights.index.unique(),RLweight.index)
# jiang_weights = jiang_weights.loc[inter_dateRange]

weight= preprocessWeight(RLweight['allocation_targets'])
chartWeight = weight
chartWeight.index = pd.to_datetime(chartWeight.index)
chartWeight = chartWeight[::-1]
# Initialize the app
app = dash.Dash(__name__)
app.title = 'Taiwan Stock Portfolio Allocation Plan'
app.config.suppress_callback_exceptions = True

def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list

   
@app.callback(
    Output('div-1', 'children'), 
    [Input(component_id='data-view', component_property='value'),
    ]
    
    )
def update_datatable(value):
    global weight    
    global tableWeight        
    if value =='Jiang':
        weight = chartWeight[::-1]
    elif value == 'RLEPR':
        weight = chartWeight[::-1] #preprocessWeight(RLweight)
    elif value == '':
        weight = chartWeight[::-1] #preprocessWeight(RLweight)

    weight= weight.reset_index().rename(columns={'index': 'Date'})
    weight.rename(columns = {'Date':f'Date \ {value}'}, inplace = True)
    if tableWeight.empty:
        tableWeight =  redesignWeight(weight)

    data = tableWeight.to_dict('records')
    max_lines = get_max_lines(data, 'Portfolio')
    cell_height = f'{20 * max_lines}px'  # Set a base height of 20px per line

    return dash_table.DataTable(
        id = "data-table",
        data=data,
        columns=[{'id': str(c), 'name': str(c)} for c in tableWeight.columns],
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        selected_columns=[],
        selected_rows=[0],  # Set the default selected row to 0
        page_action="native",
        page_current=0,
        page_size=50,
        style_table={ 
                        'maxHeight': '800px',
                        'overflow': 'scroll',
                        'width': '100%',
                        'minWidth': '100%',
                        'text-align': 'center',
                    },
                
                    fixed_rows={'headers': True, 'data': 0},
                    fixed_columns={'headers': True,'data': 1},
                    # style cell
                    style_cell={
                        'fontFamily': 'Open Sans',
                        'textAlign': 'center',
                        'height': '40px',
                        'padding': '2px 22px',
                        'whiteSpace': 'inherit',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'backgroundColor': '#31302F',
                        # 'minWidth': '200px', 'width': '200px', 'maxWidth': '200px',
                    },
                    
                    # style header
                    style_header={
                        'fontWeight': 'bold',
                        'backgroundColor': 'rgb(48, 51, 48)',
                        'maxWidth':0
                    },
                    # style filter
                    # style data
                    style_data_conditional=[
                        {
                            # stripped rows
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#888C8B'
                        },
                        {'if': {'column_id': 'Portfolio'},
                                 'width': '700px'},
                        {'if': {'column_id': 'Portfolio'},
                                'height': cell_height},
                         {'if': {'column_id': 'Add'},
                                 'width': '350px'},
                        {'if': {'column_id': 'Drop'},
                        'width': '350px'},
                        {'if': {'column_id': 'WeeklyROI'},
                        'width': '150px'},
                        {'if': {'column_id': 'ROI'},
                        'width': '150'},
                    
                    ],
                    fill_width = False

        )



@app.callback(
    Output('callBackPieGraph', 'figure'), 
    [Input("data-table", "active_cell")],
     [State("active-cell-store", "children")]
    )
def get_cell_clicked(active_cell,_):
    global weight
    
    # For pie chart
    if active_cell:
        row = active_cell['row']
        
    else:
        row=0
    pieWeightValue = weight.iloc[row]
    # print(pieWeightValue)
    asset=pieWeightValue['Asset']
    asset = '{:,.2f}'.format(float(asset))
    titlefont  = dict(family = 'Arial', color='white', size= 30)
    pieWeightValue = pieWeightValue.loc[:'Asset']
    try:
        currentDay = pieWeightValue[0].strftime('%Y-%m-%d')
    except AttributeError as e:
        currentDay = pieWeightValue.name.strftime('%Y-%m-%d')
    title=f'Portfolio Allocation <br><sup>{currentDay} Asset: ${asset} TWD </sup>'
    pieWeightValue = pieWeightValue.replace('-', '0.00')
    pieChartData = pieWeightValue.iloc[1:-1][pieWeightValue.iloc[1:-1].astype(float) > 0]
    pieDF = pieChartData.to_frame()
    pieDF['Ticker'] = pieDF.index
    pieDF=pieDF.rename(columns = {pieDF.columns[0]:'Allocation %'})

    pieFig=px.pie(pieDF,values= pieDF.columns[0],names = 'Ticker', title=title)

    pieFig.update_layout(template = 'plotly_dark',
                title = title,
                title_font= titlefont,
                width  = 430, height = 400)

    return pieFig

@app.callback(
    Output("active-cell-store", "children"),
    [Input("data-table", "active_cell")],
)
def update_active_cell_store(active_cell):
    if active_cell:
        return active_cell["row"]
    return None


@app.callback(Output('timeseries', 'figure'),
              [Input('stockselector', 'value')])
def update_graph(selected_dropdown_value):
    trace1 = []
    df_sub = df

    for stock in selected_dropdown_value:
        trace1.append(go.Scatter(x=df_sub[df_sub['Stock_ID'] == stock].index,
                                 y=df_sub[df_sub['Stock_ID'] == stock]['close'],
                                 mode='lines',
                                 opacity=0.7,
                                 name=stock,
                                 textposition='bottom center'))
    
    trace2 = go.Scatter(x=chartWeight.index,
                        y=chartWeight['WeeklyROI'],
                        mode='lines+markers',
                        opacity=1,   
                        name=' WeeklyROI',
                        yaxis='y2',
                        textposition='bottom center',
                        line=dict(dash='5,5',color='rgb(255, 255, 255)'),
                        marker=dict(symbol='circle', size=2,opacity=.5))

    trace1.append(trace2)
    data = trace1
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#6eb9ba", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                #   paper_bgcolor='rgba(0, 0, 0, 0)',
                #   plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  width=None,  
                  height=None,
                  title={'text': 'Stock Prices', 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
                  yaxis={'title': 'Stock Price'},
                  yaxis2={'title': 'WeeklyROI', 'overlaying': 'y', 'side': 'right'}
              ),
              }

    return figure

def get_max_lines(data, column):
    max_lines = 1
    for row in data:
        cell_value = row[column]
        lines = len(str(cell_value).splitlines())
        max_lines = max(max_lines, lines)
    return max_lines


def get_data_object(user_selection):
    """
    For user selections, return the relevant in-memory data frame.
    """
    return weight[user_selection]

def update_table(user_selection):
    """
    For user selections, return the relevant table
    """
    
    df = get_data_object(user_selection)
    columns = [{'name': col, 'id': col} for col in df.columns]
    data = df.to_dict(orient='records')
    return data, columns

 
app.layout = html.Div(
    children=[
        html.Div(  # Main container div
            className="row",
            children=[
                html.Div(  # Upper left side
                    className="div-user-controls",
                    style={"width": "70%", "height": "50%", "display": "inline-block"},
                    children=[
                        html.H2("STOCK PRICES \n Portfolio Allocation"),
                        html.P("Visualising time series with Plotly - Dash."),
                        html.P("Pick one or more stocks from the dropdown below."),
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                dcc.Dropdown(
                                    id="stockselector",
                                    options=get_options(df["Stock_ID"].unique()),
                                    multi=True,
                                    value=[df["close"].sort_values()[0]],
                                    style={"backgroundColor": "#A2E6EE"},
                                    className="stockselector",
                                ),
                            ],
                            style={"color": "#1E1E1E"},
                        ),
                        dcc.RadioItems(
                            id="data-view",
                            options=[
                                {"label": "RLEPR", "value": "RLEPR"},
                            ],
                            value="",
                            labelStyle={"display": "inline-block"},
                        ),
                        
                        dcc.Loading(  # Add this Loading component to wrap around the Graph
                            id="loading-timeseries",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="timeseries",
                                    config={"displayModeBar": False},
                                    animate=True,
                                    style={
                                        "width": "100%",
                                        "height": "100%",
                                        "display": "inline-block",
                                        "border": "3px #5c5c5c solid",
                                        "padding-top": "10px",
                                        "padding-left": "0px",
                                        "overflow": "hidden",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                #     ],
                # ),
                html.Div(  # Upper right side
                    className="div-for-charts bg-grey",
                    style={"width": "30%", "height": "50%", "display": "inline-block"},
                    children=[
                        dcc.Graph(
                            id="callBackPieGraph",
                            figure=get_cell_clicked({"row": 0},None),
                            style={
                                "width": "100%",
                                "height": "100%",
                                "display": "inline-block",
                                "border": "3px #5c5c5c solid",
                                "padding-top": "10px",
                                "padding-left": "0px",
                                "overflow": "hidden",
                            },
                            responsive=True,
                        ),
                    ],
                ),
            ],
        ),
        html.Div(  # Bottom part with datatable
            id="div-1",
            style={"width": "100%", "height": "50%"},
            children=[
                html.Div(style={"height": "50px"}),
                html.Div(
                    id="child-div",
                    style={
                        "width": "100%",
                        "height": "100%",
                        "border": "3px #5c5c5c solid",
                        "padding": "50px",
                        "box-sizing": "border-box",
                    },
                ),
            ],
        ),
        html.Div(id="active-cell-store", style={"display": "none"}),  # Add this line here
    ],
    style={
        "width": "100%",
        "height": "100%",
        "backgroundColor": "#808080",
    },
)


if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', port = 5000)
