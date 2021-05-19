import plotly.express as px
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras import models
from keras import layers

import numpy as np

# Load the digits dataset
digits = datasets.load_digits()

# Image reconstruction
img_shape = digits.images[-1].shape
white = np.zeros(img_shape) + 255

# Data pre-processing
enc = OneHotEncoder()
X = digits.images
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = X_train.reshape((-1, 8 * 8))
y_train = y_train.reshape(-1, 1)
y_train = enc.fit_transform(y_train).toarray()

X_test = X_test.reshape((-1, 8 * 8))
y_test = y_test.reshape(-1, 1)
y_test = enc.fit_transform(y_test).toarray()

# Model
nn = models.Sequential()
nn.add(layers.Dense(64, activation="selu", input_shape=(8 * 8,)))
nn.add(layers.Dense(64, activation="selu"))
nn.add(layers.Dense(10, activation="softmax"))
nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
nn.summary()

# Training
history = nn.fit(
    X_train, y_train, epochs=1, verbose=1, validation_data=(X_test, y_test)
)

# Training results
training_fig = go.Figure()
training_fig.add_trace(go.Scatter(y=history.history["loss"], name="training loss"))
training_fig.add_trace(
    go.Scatter(y=history.history["accuracy"], name="training accuracy")
)
training_fig.add_trace(go.Scatter(y=history.history["val_loss"], name="test loss"))
training_fig.add_trace(
    go.Scatter(y=history.history["val_accuracy"], name="test accuracy")
)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# App
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.H2("Dataset preview"),
        html.Div(id="preview"),
        html.H3("Idx: Num:", id="info"),
        dcc.Slider(
            id="num-slider",
            min=0,
            max=len(digits.images) - 1,
            step=1,
            value=0,
        ),
        html.Hr(),
        html.H2("Model results"),
        html.Div(
            dcc.Graph(figure=training_fig),
            id="train-results",
        ),
        html.Div(id="preview-demo"),
        html.H3("Idx: Num:", id="info-demo"),
        dcc.Slider(
            id="num-slider-demo",
            min=0,
            max=len(X_test) - 1,
            step=1,
            value=0,
        ),
    ]
)


@app.callback(
    [Output("preview", "children"), Output("info", "children")],
    [Input("num-slider", "value")],
)
def update_num_image(value):
    fig = px.imshow(white - digits.images[value], binary_string=True)
    return dcc.Graph(figure=fig), f"Idx: {value} Num: {digits.target[value]}"


@app.callback(
    [Output("preview-demo", "children"), Output("info-demo", "children")],
    [Input("num-slider-demo", "value")],
)
def update_num_image_demo(value):
    data = X_test[value].reshape(8, 8)
    fig = px.imshow(white - data, binary_string=True)
    label = np.argmax(y_test[value])
    pred_label = np.argmax(nn.predict(data.reshape(-1, 8 * 8))[0])
    return dcc.Graph(figure=fig), f"Idx: {value} Num: {label} Pred: {pred_label}"


app.run_server()
