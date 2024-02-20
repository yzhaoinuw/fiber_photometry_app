# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:17:50 2024

@author: yzhao
"""

import os
import base64
import tempfile
import webbrowser
from threading import Timer
import matplotlib.pyplot as plt
from io import BytesIO

import dash
from dash import Dash, html
from flask_caching import Cache
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import tdt
import numpy as np
from scipy.signal import filtfilt

import plotly.graph_objects as go
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

from components import Components


app = Dash(
    __name__,
    title="Fiber Photometry Preprocessing App",
    suppress_callback_exceptions=True,
)
components = Components()
app.layout = components.home_div
PORT = 8050

TEMP_PATH = os.path.join(tempfile.gettempdir(), "fiber_photometry_app_data")
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": TEMP_PATH,
        "CACHE_THRESHOLD": 3,
        "CACHE_DEFAULT_TIMEOUT": 86400,  # to save cache for 1 day, otherwise it is default to 300 seconds
    },
)


def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{PORT}/")


# %% callbacks


@app.callback(Output("debug-container", "children"), Input("file-explorer", "value"))
def show_state(path):
    return path


@app.callback(
    Output("selection-container", "children"),
    Output("visualization-container", "children", allow_duplicate=True),
    Input("read-button", "n_clicks"),
    State("file-explorer", "value"),
    prevent_initial_call=True,
)
def list_channels(n_clicks, folder_path):
    if folder_path is None:
        raise PreventUpdate
    # Load the FP data
    data = tdt.read_block(folder_path)
    cache.set("data", data)

    # display the list of channel names in data.streams
    channel_list = data.streams
    select_channel = [
        {"value": channel_name, "label": channel_name}
        for channel_name in channel_list.keys()
    ]
    select_exp_component = dmc.Select(
        label="Select exp data channel",
        placeholder="",
        id="exp-data-select",
        value="",
        data=select_channel,
    )
    select_control_component = dmc.Select(
        label="Select control data channel",
        placeholder="",
        id="control-data-select",
        value="",
        data=select_channel,
    )

    pulse_list = data.epocs
    select_pulse = [
        {"value": pulse_name, "label": pulse_name} for pulse_name in pulse_list.keys()
    ]
    select_pulse_component = dmc.Select(
        label="Select pulse",
        placeholder="",
        id="pulse-select",
        value="",
        data=select_pulse,
    )

    enter_filter_component = dmc.NumberInput(
        id="filter-input",
        label="Enter mean filter value",
        # description="",
        value=1000,
        min=1,
        max=2000,
        # style={"width": 250},
    )

    return [
        select_exp_component,
        select_control_component,
        select_pulse_component,
        enter_filter_component,
    ], []


@app.callback(
    Output("lower-container", "children"),
    Input("exp-data-select", "value"),
    Input("control-data-select", "value"),
    Input("pulse-select", "value"),
    prevent_initial_call=True,
)
def show_process_button(exp_channel, control_channel, pulse_channel):
    if exp_channel is None or control_channel is None or pulse_channel is None:
        return dash.no_update

    return html.Button(
        ["Fit Data"],
        id="process-button",
    )


@app.callback(
    Output("visualization-container", "children", allow_duplicate=True),
    Input("process-button", "n_clicks"),
    State("exp-data-select", "value"),
    State("control-data-select", "value"),
    State("pulse-select", "value"),
    State("filter-input", "value"),
    prevent_initial_call=True,
)
def fit_data(n_clicks, exp_channel, control_channel, ttl_pulse, filter_value):
    if n_clicks is None:
        raise PreventUpdate

    data = cache.get("data")
    # analysis pipeline below
    signal_fs = data.streams[exp_channel].fs
    signal_465 = data.streams[exp_channel].data  # Assuming data is 1D
    signal_405 = data.streams[control_channel].data  # Assuming data is 1D

    # Removing FP trace prior to first TTL pulse
    TTL_FP = data.epocs[ttl_pulse].onset

    TTL_gap = np.diff(TTL_FP) > 10
    if np.any(TTL_gap):
        first_gap_index = np.where(TTL_gap)[0][0]
        TTL_onset = TTL_FP[first_gap_index + 1]
    else:
        TTL_onset = TTL_FP[0]

    first_TTL = int(TTL_onset * signal_fs)
    signal_465 = signal_465[first_TTL:]
    signal_405 = signal_405[first_TTL:]

    # Normalize and plot
    MeanFilterOrder = filter_value
    MeanFilter = np.ones(MeanFilterOrder) / MeanFilterOrder
    fs_signal = np.arange(len(signal_465))
    sec_signal = fs_signal / signal_fs
    reg = np.polyfit(signal_405, signal_465, 1)
    a, b = reg
    controlFit_465 = a * signal_405 + b
    controlFit_465 = filtfilt(MeanFilter, 1, controlFit_465)
    normDat = (signal_465 - controlFit_465) / controlFit_465
    delta_465 = normDat * MeanFilterOrder
    cache.set("time", sec_signal[1000:])
    cache.set("normalized_signal", delta_465[1000:])

    rmse = np.sqrt(np.mean((signal_465[1000:] - controlFit_465[1000:]) ** 2))
    components.rmse_label.children = f"RMSE: {rmse:.2f}"

    fig = plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(sec_signal[1000:], signal_405[1000:])
    plt.title("raw control")
    plt.subplot(4, 1, 2)
    plt.plot(sec_signal[1000:], signal_465[1000:])
    plt.title("raw signal")
    plt.subplot(4, 1, 3)
    plt.plot(sec_signal[1000:], signal_465[1000:])
    plt.plot(sec_signal[1000:], controlFit_465[1000:])
    plt.title("fitted control")
    plt.subplot(4, 1, 4)
    plt.plot(sec_signal[1000:], delta_465[1000:])
    plt.title("normalized signal")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    components.matplotlib_image.src = f"data:image/png;base64,{fig_data}"
    # components.confirm_button.style = {"display": "block"}
    return components.image_div


@app.callback(
    Output("confirm-button", "style"),
    Input("fitting-feedback-dropdown", "value"),
    prevent_initial_call=True,
)
def show_confirm_button(value):
    return {"display": "block"}


@app.callback(
    Output("visualization-container", "children", allow_duplicate=True),
    Input("confirm-button", "n_clicks"),
    State("fitting-feedback-dropdown", "value"),
    prevent_initial_call=True,
)
def show_visualization(n_clicks, value):
    if value != "Yes":
        return dash.no_update

    sec_signal = cache.get("time")
    delta_465 = cache.get("normalized_signal")
    fig = FigureResampler(
        go.Figure(),
        default_n_shown_samples=2000,
        default_downsampler=MinMaxLTTB(parallel=True),
    )
    fig.add_trace(
        go.Scattergl(
            name="normalized signal",
            line=dict(width=1),
            marker=dict(size=2),
            showlegend=False,
            mode="lines+markers",
            hoverinfo="x+y",
        ),
        hf_x=sec_signal,
        hf_y=delta_465,
    )

    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=20, r=20, b=40),
        height=800,
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
        title_text="Normalized Signal",
        font=dict(
            size=12,  # title font size
        ),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="pan",
        # clickmode="event",
    )
    cache.set("fig_resampler", fig)
    components.graph.figure = fig
    return components.graph


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("graph", "relayoutData"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata):
    fig = cache.get("fig_resampler")
    return fig.construct_update_data_patch(relayoutdata)


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=PORT)
