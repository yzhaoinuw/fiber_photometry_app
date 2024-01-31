# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:53:23 2024

@author: yzhao
"""

from dash import dcc, html

from util import FileTree


home_div = html.Div(
    [
        html.Div([FileTree('C:/Users/yzhao/python_projects/fiber_photometry/Datafiles_Viewpoint_TuckerDavis/TDT files').render()],
            id="upper-container"),
        html.Div([html.Button(
            ["Read Data"],
            id="read-button",
        )],
        id="middle-container"),
        html.Div(id="debug-container"),
        #html.Div(id="exp-channel-container"),
        #html.Div(id="control-channel-container"),
        #html.Div(id="ttl-pulse-container"),
        html.Div(id="selection-container"),
        html.Div(id="lower-container"),
        html.Div(
            [html.Img(id='figure-image')],
            id="visualization-container"),
    ]
)