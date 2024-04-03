# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:39:55 2024

@author: yzhao
"""

import tdt
import numpy as np
from scipy import stats
from scipy.signal import filtfilt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB


def extract_FP_data(data, exp_data_channel, control_data_channel, ttl_pulse=None):
    # read the FP data
    signal_fs = data.streams[exp_data_channel].fs
    signal_465 = data.streams[exp_data_channel].data  # Assuming data is 1D
    signal_405 = data.streams[control_data_channel].data  # Assuming data is 1D

    if ttl_pulse is not None:
        # Removing FP trace prior to first TTL pulse
        TTL_FP = data.epocs[ttl_pulse].onset

        TTL_gap = np.diff(TTL_FP) > 10  # should it be 6, or 5?
        if np.any(TTL_gap):
            first_gap_index = np.where(TTL_gap)[0][0]
            TTL_onset = TTL_FP[first_gap_index + 1]
        else:
            TTL_onset = TTL_FP[0]

        first_TTL = int(TTL_onset * signal_fs)
        signal_465 = signal_465[first_TTL:]
        signal_405 = signal_405[first_TTL:]
    return signal_fs, signal_465, signal_405


# %%
def filt_signal(signal_465, signal_405, MeanFilterOrder=1000):
    MeanFilter_2 = np.ones(MeanFilterOrder) / MeanFilterOrder
    MeanFilter = np.ones(1000) / 1000
    filt_465 = filtfilt(MeanFilter_2, 1, signal_465)
    filt_405 = filtfilt(MeanFilter, 1, signal_405)
    return filt_465, filt_405


def fit_data(signal_fs, filt_465, filt_405):
    fs_signal = np.arange(len(filt_465))
    time_second = fs_signal / signal_fs
    reg = np.polyfit(filt_405, filt_465, 1)
    a, b = reg
    controlFit_465 = a * filt_405 + b
    deltaF_over_F = (filt_465 - controlFit_465) / controlFit_465
    deltaF_over_F_percentage = deltaF_over_F * 100  # convert to %

    z_score = stats.zscore(deltaF_over_F)
    return time_second, controlFit_465, deltaF_over_F_percentage, z_score


# %%
def make_figure(
    time_second, deltaF_over_F_percentage, normalized_z_score, cutoff_seconds=5
):
    signal_fs = 1 / time_second[1]
    cutoff = round(signal_fs * cutoff_seconds)
    hours = np.arange(
        0, np.ceil(time_second[-1] / 3600) + 1, 0.5
    )  # Calculate the hour marks
    tickvals = hours * 3600  # Convert hours to seconds for positioning
    ticktext = [f"{hour}h" for hour in hours]

    fig = FigureResampler(
        make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "Delta_F / F",
                "Z Score Signal",
            ),
            row_heights=[0.5, 0.5],
        ),
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
            hovertemplate="<b>time</b>: %{x:.2f}s"
            + "<br><b>y</b>: %{y:.2f}<extra></extra>",
        ),
        hf_x=np.copy(time_second[cutoff:]),
        hf_y=np.copy(deltaF_over_F_percentage[cutoff:]),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            line=dict(width=1),
            marker=dict(size=2),
            showlegend=False,
            mode="lines+markers",
            hovertemplate="<b>time</b>: %{x:.2f}s"
            + "<br><b>y</b>: %{y:.2f}<extra></extra>",
        ),
        hf_x=np.copy(time_second[cutoff:]),
        hf_y=np.copy(normalized_z_score[cutoff:]),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=800,
        hovermode="x unified",  # gives crosshair in one subplot
        yaxis_title="percentage",
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="pan",
    )
    fig.update_traces(xaxis="x2")

    fig.update_xaxes(
        tickcolor="black",
        tickvals=tickvals,  # Set custom tick values (positions)
        ticktext=ticktext,
        minor=dict(
            ticks="outside",
            dtick=60,
            tick0=time_second[cutoff],
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        range=[0, np.ceil(time_second[-1])],
        row=2,
        col=1,
        title_text="<b>Time</b>",
    )
    return fig


# %%

if __name__ == "__main__":
    FP_data_path = "./Datafiles_Viewpoint_TuckerDavis/TDT files/20220809_DPH_588_592"
    exp_data_channel = "_465C"
    control_data_channel = "_405C"
    ttl_pulse = "Pu1_"

    fp_data = tdt.read_block(FP_data_path)
    signal_fs, signal_465, signal_405 = extract_FP_data(
        fp_data, exp_data_channel, control_data_channel, ttl_pulse=None
    )
    filt_465, filt_405 = filt_signal(signal_465, signal_405, MeanFilterOrder=1000)
    time_second, _, deltaF_over_F_percentage, normalized_z_score = fit_data(
        signal_fs, filt_465, filt_405
    )
    fig = make_figure(time_second, deltaF_over_F_percentage, normalized_z_score)
    fig.show_dash(config={"scrollZoom": True})
