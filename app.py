from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State
import dash_uploader as du
import os
from flask import send_file
import sys
import base64
#imports
import math
import pandas as pd
import numpy as np
import pickle
import easyocr

from sklearn.preprocessing import LabelEncoder
encodeX = LabelEncoder()
encodeY = LabelEncoder()

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

from sklearn.naive_bayes import GaussianNB

df_test = pd.read_csv('training\\df_test.csv')
df_test = df_test.drop('Unnamed: 0', axis=1)

savFile = 'training\\finalized_model.sav'
loaded_model = pickle.load(open(savFile, 'rb'))

def inputPredict(Gender, Married, Education, Self_Employed,
                  ApplicantIncome, CoapplicantIncome, LoanAmount,
                  Loan_Amount_Term, Credit_History):

  Dependents = '2'
  Property_Area = 'Rural'
  LoanAmount_log = math.log(LoanAmount)
  TotalIncome = ApplicantIncome + CoapplicantIncome
  TotalIncome_log = math.log(TotalIncome)

  df = df_test
  df.loc[-1] = [0, Gender, Married, Dependents, Education, Self_Employed,
                  ApplicantIncome, CoapplicantIncome, LoanAmount,
                  Loan_Amount_Term, Credit_History, Property_Area, LoanAmount_log,
                TotalIncome, TotalIncome_log]

  test = df.iloc[:, np.r_[1:5,9:11,13:15]].values

  for k in range(0,5):
    test[:,k] = encodeX.fit_transform(test[:,k])

  test[:,7] = encodeY.fit_transform(test[:,7])

  test=ss.fit_transform(test)

  prediction = loaded_model.predict(test)
  return prediction[-1]

#filepath refers to the png path
def OCR(filepath):
  reader = easyocr.Reader(['en'])
  result = reader.readtext(filepath)

  list = []

  for(bbox, text, prob) in result:
    list.append(text)

  gender = list[9]
  maritalStatus = list[11]
  education = list[13]
  applicantIncome = list[33]
  coapplicantIncome = list[40]
  selfEmployed = list[35]
  loanAmount = list[47]
  loanTerm = list[49]
  creditHistory = list[53]
  
  creditProvided = 0.0

  if creditHistory == 'Yes':
    creditProvided = 1.0
  else:
    creditProvided = 0.0

  pred = inputPredict(gender, maritalStatus, education, selfEmployed, float(applicantIncome)/100, float(coapplicantIncome)/100, float(loanAmount), float(loanTerm), creditProvided)
  return pred

# Initialize the Dash app
app = Dash(__name__, suppress_callback_exceptions=True)
du.configure_upload(app, folder='uploaded_files')

# Custom HTML layout to include Google Font
app.index_string = ''' 
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        {%css%}
    </head>
    <body style="margin: 0; padding: 0; overflow: hidden; background-color: #010F33;">  <!-- Change the background color here -->
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

nav_link_style = {
    'padding': '10px 15px',
    'color': 'white',
    'textDecoration': 'none',
    'borderRadius': '5px',
    'transition': 'background-color 0.3s',
    'margin': '0 10px',
    'backgroundColor': 'black',
    'cursor': 'pointer',
}

centered_button_style = {
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center',
    'height': 'calc(100vh - 40px)',
    'background-color': 'black',
    'marginRight': '5px',
}
# Define the index page layout
def index_page():
    nav_link_style = {
        'padding': '10px 15px',
        'color': 'white',
        'textDecoration': 'none',
        'borderRadius': '5px',
        'transition': 'background-color 0.3s',
        'margin': '0 10px',  
        'backgroundColor': 'black',  
        'cursor': 'pointer',
    }

    centered_button_style = {
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'height': 'calc(100vh - 40px)',
        'background-color': 'black',
        'marginRight': '5px',
    }

    return html.Div([
        html.Div([
            dcc.Link('Home', href='/', style=nav_link_style),
            dcc.Link('Upload File', href='/upload', style=nav_link_style),
            ], style={'backgroundColor': 'white', 'padding': '10px 0', 'display': 'flex', 'justifyContent': 'center'}),

        html.Div(style={
            'height': 'calc(100vh - 40px)',  
            'backgroundImage': 'url("/assets/HomePage.png")',
            'backgroundSize': 'cover',
            'backgroundPosition': 'center center',
            'backgroundRepeat': 'no-repeat',
            'position': 'relative',
            'overflow': 'hidden'
        }),

        html.Div([
            html.Div([
                dcc.Link('Home', href='/', style={'color': 'white'}),
            ], style=centered_button_style),

            html.Div([
                dcc.Link('Upload File', href='/upload', style={'color': 'white'}),
            ], style=centered_button_style),

            html.Div([
                dcc.Link('Manual Questions', href='/manual-questions', style={'color': 'white'})  # Add a link to the manual questions page
            ], style=centered_button_style),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'gap': '20px'}),
    ])

# Define the upload page layout
def upload_page():
    nav_link_style = {
        'padding': '10px 15px',
        'color': 'white',
        'textDecoration': 'none',
        'borderRadius': '5px',
        'transition': 'background-color 0.3s',
        'margin': '0 10px',
        'backgroundColor': 'black',
        'cursor': 'pointer',
    }

    centered_button_style = {
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'height': 'calc(100vh - 100px)',
        'background-color': 'black',
        'marginRight': '5px',
    }

    drag_and_drop_area = dcc.Upload(
        id="upload-png",
       
        multiple=False,
        style={
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'padding': '5x',
            'textAlign': 'center',
            'margin': '0 auto',
            'height': ' 135px',  # Adjusted height
            'lineHeight': '35px',  # Adjusted line height
            'backgroundColor': 'rgba(255, 255, 255, 0.2)',
        }
    )

    return html.Div([
        html.Div([
            dcc.Link('Home', href='/', style=nav_link_style),
            dcc.Link('Upload File', href='/upload', style=nav_link_style)
            ], style={'backgroundColor': 'white', 'padding': '10px 0', 'display': 'flex', 'justifyContent': 'center'}),

        html.Div(style={
            'height': 'calc(100vh - 40px)',
            'backgroundImage': 'url("/assets/file.png")',
            'backgroundSize': 'cover',
            'backgroundPosition': 'center center',
            'backgroundRepeat': 'no-repeat',
            'position': 'relative',
            'overflow': 'hidden'
        }),

        html.Div([
            html.Div([
                drag_and_drop_area,
                html.Div(id='file-output')
            ], style={'backgroundColor': 'rgba(255, 255, 255, 0)', 'borderRadius': '5px', 'padding': '5px', 'textAlign': 'center', 'position': 'absolute', 'bottom': '300px', 'left': '50%', 'transform': 'translateX(-50%)', 'width': ' 17%'}),
        ]),
    ])


# Add the manual questions page layout
def manual_questions_page():
    nav_link_style = {
        'padding': '10px 15px',
        'color': 'white',
        'textDecoration': 'none',
        'borderRadius': '5px',
        'transition': 'background-color 0.3s',
        'margin': '0 10px',
        'backgroundColor': 'black',
        'cursor': 'pointer',
    }

    centered_button_style = {
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'height': 'calc(100vh - 40px)',
        'background-color': 'black',
        'marginRight': '5px',
    }

    # Define the options for the manual questions
    gender_options = ['male', 'female']
    married_options = ['unmarried', 'married']
    education_options = ['Graduate', 'Not Graduate']
    self_employed_options = ['Yes', 'No']
    credit_history_options = ['1', '0']
    property_area_options = ['Urban', 'Rural', 'Semiurban']

    # Define the left side with the logo
    left_side = html.Div([
        html.Img(src='/assets/logo.jpg', style={'width': '110%'}),  # Add the logo and scale it up by 110%
    ], style={
        'width': '50%',
        'height': 'calc(100vh - 40px)',
        'backgroundPosition': 'center center',
        'backgroundRepeat': 'no-repeat',
        'position': 'relative',
        'overflow': 'hidden',
        'float': 'left',
    })

    # Define the right side with the questions
    right_side = html.Div([
        html.H1('Manual Questions', style={'color': 'white', 'textAlign': 'center'}),
        html.Div([
            html.Label('Gender', style={'color': 'white'}),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[{'label': gender, 'value': gender} for gender in gender_options],
                style={'width': '100%'},
            ),
            html.Label('Married', style={'color': 'white'}),
            dcc.Dropdown(
                id='married-dropdown',
                options=[{'label': married, 'value': married} for married in married_options],
                style={'width': '100%'},
            ),
            html.Label('Education', style={'color': 'white'}),
            dcc.Dropdown(
                id='education-dropdown',
                options=[{'label': education, 'value': education} for education in education_options],
                style={'width': '100%'},
            ),
            html.Label('Self Employed', style={'color': 'white'}),
            dcc.Dropdown(
                id='self-employed-dropdown',
                options=[{'label': self_employed, 'value': self_employed} for self_employed in self_employed_options],
                style={'width': '100%'},
            ),
            html.Label('Credit History', style={'color': 'white'}),
            dcc.Dropdown(
                id='credit-history-dropdown',
                options=[{'label': credit_history, 'value': credit_history} for credit_history in credit_history_options],
                style={'width': '100%'},
            ),
            html.Label('Property Area', style={'color': 'white'}),
            dcc.Dropdown(
                id='property-area-dropdown',
                options=[{'label': property_area, 'value': property_area} for property_area in property_area_options],
                style={'width': '100%'},
            ),
            html.Button('Submit', id='submit-button', n_clicks=0, style={'margin-top': '20px'}),
            html.Div(id='manual-questions-output', style={'color': 'white', 'margin-top': '-20px'})
        ], style={'max-width': '400px', 'margin': '0 auto'})
    ], style={
        'width': '50%',
        'height': 'calc(100vh - 40px)',
        'backgroundColor': '#001D37',  # Change the background color here
        'color': 'white',
        'position': 'relative',
        'overflow': 'hidden',
        'float': 'right',
    })

    return html.Div([
        html.Div([
            dcc.Link('Home', href='/', style=nav_link_style),
            dcc.Link('Upload File', href='/upload', style=nav_link_style),
            dcc.Link('Manual Questions', href='/manual-questions', style=nav_link_style)
        ], style={'backgroundColor': 'white', 'padding': '10px 0', 'display': 'flex', 'justifyContent': 'center'}),

        html.Div([
            left_side,  # Display the left side with the logo
            right_side,  # Display the right side with the questions
        ])
    ])


congratulations_layout = html.Div([
        html.Div([
            dcc.Link('Home', href='/', style=nav_link_style),
            dcc.Link('Upload File', href='/upload', style=nav_link_style)
            ], style={'backgroundColor': 'white', 'padding': '10px 0', 'display': 'flex', 'justifyContent': 'center'}),

        html.Div(style={
            'height': 'calc(100vh - 40px)',  
            'backgroundImage': 'url("assets/congratulations_background.jpg")',
            'backgroundSize': 'cover',
            'backgroundPosition': 'center center',
            'backgroundRepeat': 'no-repeat',
            'position': 'relative',
            'overflow': 'hidden'
        }),

        html.Div([
            html.Div([
                dcc.Link('Home', href='/', style={'color': 'white'}),
            ], style=centered_button_style),

            html.Div([
                dcc.Link('Upload File', href='/upload', style={'color': 'white'}),
            ], style=centered_button_style),

            html.Div([
                dcc.Link('Manual Questions', href='/manual-questions', style={'color': 'white'})  # Add a link to the manual questions page
            ], style=centered_button_style),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'gap': '20px'}),
    ])

# Define the layout for the sorry page
sorry_layout = html.Div([
        html.Div([
            dcc.Link('Home', href='/', style=nav_link_style),
            dcc.Link('Upload File', href='/upload', style=nav_link_style),
            ], style={'backgroundColor': 'white', 'padding': '10px 0', 'display': 'flex', 'justifyContent': 'center'}),

        html.Div(style={
            'height': 'calc(100vh - 40px)',  
            'backgroundImage': 'url("assets/sorry_background.png")',
            'backgroundSize': 'cover',
            'backgroundPosition': 'center center',
            'backgroundRepeat': 'no-repeat',
            'position': 'relative',
            'overflow': 'hidden'
        }),

        html.Div([
            html.Div([
                dcc.Link('Home', href='/', style={'color': 'white'}),
            ], style=centered_button_style),

            html.Div([
                dcc.Link('Upload File', href='/upload', style={'color': 'white'}),
            ], style=centered_button_style),

            html.Div([
                dcc.Link('Manual Questions', href='/manual-questions', style={'color': 'white'})  # Add a link to the manual questions page
            ], style=centered_button_style),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'gap': '20px'}),
    ])



# Define the layout of the app
app.layout = html.Div([ 
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
])

# Callback for page content
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def render_page_content(pathname):
    if pathname == '/upload':
        return upload_page()
    elif pathname == '/manual-questions':
        return manual_questions_page()
    elif pathname == "/congratulations":
        return congratulations_layout
    elif pathname == "/sorry":
        return sorry_layout
    else:
        return index_page()

@app.callback(Output("url", "pathname"),
              [Input("upload-png", 'contents')],
              [State("upload-png", "filename")])
def update_output(contents, filename):
    if contents is not None:
        # Save the uploaded file to a folder
        file_path = os.path.join('C:\\Users\\chara\\OneDrive\\Desktop\\upfiles\\upfiles', filename)
        values = base64.b64decode(contents.split(",")[1])
        with open(file_path, 'wb') as f:
            f.write(values) 
        print(file_path)
        output = OCR(file_path)
        if (output == 1):
            return "congratulations"
        else:
           return "sorry"

    else:
        return None


# Callback to handle manual questions submission
@app.callback(
    Output('manual-questions-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('gender-dropdown', 'value'),
     State('married-dropdown', 'value'),
     State('education-dropdown', 'value'),
     State('self-employed-dropdown', 'value'),
     State('credit-history-dropdown', 'value'),
     State('property-area-dropdown', 'value')]
)
def handle_manual_questions_submission(n_clicks, gender, married, education, self_employed, credit_history, property_area):
    if n_clicks > 0:
        return f'Gender: {gender}, Married: {married}, Education: {education}, Self Employed: {self_employed}, Credit History: {credit_history}, Property Area: {property_area}'

if __name__ == '__main__':
    app.run_server(debug=True)
