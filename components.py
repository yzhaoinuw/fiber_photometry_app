# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:53:23 2024

@author: yzhao
"""

from dash import html, dcc
import dash_mantine_components as dmc

from util import FileTree


graph = dcc.Graph(
    id="graph",
    config={
        "scrollZoom": True,
        "editable": True,
        "edits": {
            "axisTitleText": False,
            "titleText": False,
            "colorbarTitleText": False,
            "annotationText": False,
        },
    },
    style={"width": "80%"},
)

matplotlib_image = html.Img(id="figure-image")
confirm_button = html.Button(
    ["Confirm"], id="confirm-button", style={"display": "none"}
)
rmse_label = dmc.Text(id="rmse-label")
rsquared_label = dmc.Text(id="rsquared-label")

image_div = html.Div(
    style={"display": "flex"},
    children=[
        matplotlib_image,
        html.Div(
            [
                rmse_label,
                rsquared_label,
                dmc.Select(
                    label="Does the fitting look good?",
                    placeholder="Yes/No",
                    id="fitting-feedback-dropdown",
                    value="",
                    data=[
                        {"value": "Yes", "label": "Yes"},
                        {"value": "No", "label": "No"},
                    ],
                    # style={"display": "none"}
                ),
                confirm_button,
            ]
        ),
    ],
)

home_div = html.Div(
    [
        html.Div(
            [
                FileTree(
                    "./TDT_files"
                ).render()
            ],
            id="upper-container",
        ),
        html.Div(
            [
                html.Button(
                    ["Read Data"],
                    id="read-button",
                )
            ],
            id="middle-container",
        ),
        html.Div(id="debug-container"),
        html.Div(id="selection-container"),
        html.Div(id="lower-container"),
        html.Div(
            style={"display": "flex"},
            id="visualization-container",
        ),
    ]
)


# %%
class Components:
    def __init__(self):
        self.home_div = home_div
        self.graph = graph
        self.matplotlib_image = matplotlib_image
        self.rmse_label = rmse_label
        self.rsquared_label = rsquared_label
        self.confirm_button = confirm_button
        self.image_div = image_div
