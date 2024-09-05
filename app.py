# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:17:50 2024

@author: yzhao
"""

import os
import base64
import tempfile
import webbrowser
from io import BytesIO
from threading import Timer

import dash
from dash import Dash, html

from flask_caching import Cache
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import tdt
import numpy as np
import matplotlib.pyplot as plt

from components import Components
from func import extract_FP_data, filt_signal, fit_data, make_figure


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
    select_pulse = [{"value": None, "label": "NA (Not Available)"}]
    select_pulse.extend(
        [{"value": pulse_name, "label": pulse_name} for pulse_name in pulse_list.keys()]
    )
    select_pulse_component = dmc.Select(
        label="Select ttl pulse",
        placeholder="",
        id="pulse-select",
        value=None,
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
    if exp_channel is None or control_channel is None:
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
def plot_fit(n_clicks, exp_channel, control_channel, ttl_pulse, filter_value):
    if n_clicks is None:
        raise PreventUpdate

    data = cache.get("data")
    # analysis pipeline below
    signal_fs, signal_465, signal_405 = extract_FP_data(
        data, exp_channel, control_channel, ttl_pulse=ttl_pulse
    )
    # apply filtfilt to 465 and 405
    MeanFilterOrder = filter_value
    filt_465, filt_405 = filt_signal(
        signal_465, signal_405, MeanFilterOrder=MeanFilterOrder
    )

    time_second, controlFit_465, deltaF_over_F_percentage, normalized_z_score = (
        fit_data(signal_fs, filt_465, filt_405)
    )

    signal_fs = 1 / time_second[1]
    cutoff = round(signal_fs * 5)
    cache.set("time", time_second)
    cache.set("deltaF_over_F_percentage", deltaF_over_F_percentage)
    cache.set("normalized_z_score", normalized_z_score)

    rmse = np.sqrt(np.mean((signal_465[cutoff:] - controlFit_465[cutoff:]) ** 2))
    components.rmse_label.children = f"RMSE: {rmse:.2f}"

    fig = plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(time_second[cutoff:], filt_405[cutoff:])
    plt.title("filt control")
    plt.subplot(4, 1, 2)
    plt.plot(time_second[cutoff:], filt_465[cutoff:])
    plt.title("filt signal")
    plt.subplot(4, 1, 3)
    plt.plot(time_second[cutoff:], filt_465[cutoff:])
    plt.plot(time_second[cutoff:], controlFit_465[cutoff:])
    plt.title("fitted control")
    plt.subplot(4, 1, 4)
    plt.plot(time_second[cutoff:], deltaF_over_F_percentage[cutoff:])
    plt.title("delta F over F (%)")
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

    time_second = cache.get("time")
    deltaF_over_F_percentage = cache.get("deltaF_over_F_percentage")
    normalized_z_score = cache.get("normalized_z_score")
    fig = make_figure(time_second, deltaF_over_F_percentage, normalized_z_score)
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
    app.run_server(debug=False, port=PORT)
