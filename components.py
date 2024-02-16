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
)

matplotlib_image = html.Img(id="figure-image")
confirm_button = html.Button(
    ["Confirm"], id="confirm-button", style={"display": "none"}
)

image_div = html.Div(
    style={"display": "flex"},
    children=[
        matplotlib_image,
        html.Div(
            [
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
                    "C:/Users/yzhao/python_projects/fiber_photometry/Datafiles_Viewpoint_TuckerDavis/TDT files"
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
        self.confirm_button = confirm_button
        self.image_div = image_div
