# -*- coding: utf-8 -*-
import os
import yfinance as yf
import numpy as np
from scipy.signal import argrelextrema


def download_stocks_df(tickers=["TSLA", "FB", "MSFT"]):
    tsla_df = yf.download(tickers, auto_adjust=True)
    return tsla_df



def trading_figures_position(y, fig_name):
    maxima, minima = argrelextrema(np.array(y), np.greater)[0], argrelextrema(np.array(y), np.less)[0]
    extrema_ind = np.concatenate((maxima, minima), axis=0)
    extrema_ind = np.sort(extrema_ind)
    min_max = np.diff(np.sign(np.diff(y)))
    result = []
    ext_1 = []
    ext_2 = []
    ext_3 = []
    ext_4 = []
    ext_5 = []
    if fig_name == "H&S":
        hs = []
        for index, value in enumerate(np.nditer(extrema_ind)):
            if index + 4 > len(extrema_ind) - 1:
                break
            if index - 1 == -1 or index < 14:
                continue
            average = (y[extrema_ind[index + 4]] + y[value]) / 2
            average_1 = (y[extrema_ind[index + 3]] + y[extrema_ind[index + 1]]) / 2
            if value in maxima and extrema_ind[index + 2] in maxima and extrema_ind[index + 4] in maxima \
                    and y[extrema_ind[index + 2]] > y[extrema_ind[index + 4]] and y[extrema_ind[index + 2]] > y[value] \
                    and abs(average - y[value]) <= 0.005 * average \
                    and abs(average - y[extrema_ind[index + 4]]) <= 0.005 * average \
                    and abs(average_1 - y[extrema_ind[index + 1]]) <= 0.005 * average_1 \
                    and abs(average_1 - y[extrema_ind[index + 3]]) <= 0.005 * average_1:
                hs.append(value.item())
                ext_1.append(extrema_ind[index])
                ext_2.append(extrema_ind[index + 1])
                ext_3.append(extrema_ind[index + 2])
                ext_4.append(extrema_ind[index + 3])
                ext_5.append(extrema_ind[index + 4])
        return (hs, ext_1, ext_2, ext_3, ext_4, ext_5)
    elif fig_name == "IH&S":
        ihs = []
        for index, value in enumerate(np.nditer(extrema_ind)):
            if index + 4 > len(extrema_ind) - 1:
                break
            if index - 1 == -1 or index < 14:
                continue
            average = (y[extrema_ind[index + 4]] + y[value]) / 2
            average_1 = (y[extrema_ind[index + 3]] + y[extrema_ind[index + 1]]) / 2
            if value in minima and extrema_ind[index + 2] in minima and extrema_ind[index + 4] in minima \
                    and y[extrema_ind[index + 2]] < y[extrema_ind[index + 4]] and y[extrema_ind[index + 2]] < y[value] \
                    and abs(average - y[value]) <= 0.005 * average \
                    and abs(average - y[extrema_ind[index + 4]]) <= 0.005 * average \
                    and abs(average_1 - y[extrema_ind[index + 1]]) <= 0.005 * average_1 \
                    and abs(average_1 - y[extrema_ind[index + 3]]) <= 0.005 * average_1:
                ihs.append(value.item())
                ext_1.append(extrema_ind[index])
                ext_2.append(extrema_ind[index+1])
                ext_3.append(extrema_ind[index+2])
                ext_4.append(extrema_ind[index+3])
                ext_5.append(extrema_ind[index+4])
                
        return (ihs, ext_1, ext_2, ext_3, ext_4, ext_5)
    elif fig_name == "BTOP":
        print("BTOP")
        btop = []
        for index, value in enumerate(np.nditer(extrema_ind)):
            if index + 4 > len(extrema_ind) - 1:
                break
            if index - 1 == -1 or index < 14:
                continue

            if value in maxima and y[extrema_ind[index + 4]] > y[extrema_ind[index + 2]] > y[extrema_ind[index]] \
                    and y[extrema_ind[index + 1]] > y[extrema_ind[index+3]]:
                btop.append(value.item())
                ext_1.append(extrema_ind[index])
                ext_2.append(extrema_ind[index + 1])
                ext_3.append(extrema_ind[index + 2])
                ext_4.append(extrema_ind[index + 3])
                ext_5.append(extrema_ind[index + 4])
        return (btop, ext_1, ext_2, ext_3, ext_4, ext_5)
    elif fig_name == "BBOT":
        print("BBOT")
        bbot = []
        for index, value in enumerate(np.nditer(extrema_ind)):
            if index + 4 > len(extrema_ind) - 1:
                break
            if index - 1 == -1 or index < 14:
                continue

            if value in minima and y[extrema_ind[index + 4]] < y[extrema_ind[index + 2]] < y[extrema_ind[index]] \
                    and y[extrema_ind[index + 1]] < y[extrema_ind[index+3]]:
                bbot.append(value.item())
                ext_1.append(extrema_ind[index])
                ext_2.append(extrema_ind[index + 1])
                ext_3.append(extrema_ind[index + 2])
                ext_4.append(extrema_ind[index + 3])
                ext_5.append(extrema_ind[index + 4])
        return (bbot, ext_1, ext_2, ext_3, ext_4, ext_5)


def one_instance_candlestick(df, name, border_1, border_2, figur):
    dt = pd.DataFrame([])
    # earlier: [i - 59 + d:i + d + 2]
    print(border_1, border_2)

    dt["Close"] = list(df["Close"][border_1 - 5:border_2 + 6])
    dt["Open"] = list(df["Open"][border_1 - 5:border_2 + 6])
    dt["High"] = list(df["High"][border_1 - 5:border_2 + 6])
    dt["Low"] = list(df["Low"][border_1 - 5:border_2 + 6])
    dt["Date"] = list(df["Date"][border_1 - 5:border_2 + 6])
    dt["Max"] = max(list(dt["Open"])+list(dt["Close"]))
    dt["Min"] = min(list(dt["Open"])+list(dt["Close"]))
    print(dt["Close"])
    dt.reset_index(inplace=True)
    xaxis_dt_format = "%d %b %Y"
    print(dt["Date"])
    if dt["Date"][0].hour > 0:
        xaxis_dt_format = "%d %b %Y, %H:%M:%S"

    fig = figure(sizing_mode="stretch_both",
                 tools="xpan,xwheel_zoom,reset,save",
                 active_drag="xpan",
                 active_scroll="xwheel_zoom",
                 x_axis_type="linear",
                 title="title"
                 )
    fig.yaxis[0].formatter = NumeralTickFormatter(format="$5.3f")
    inc = dt.Close > dt.Open
    dec = ~inc

    # Colour scheme for increasing and descending candles
    INCREASING_COLOR = "#17BECF"
    DECREASING_COLOR = "#7F7F7F"

    width = 0.5
    inc_source = ColumnDataSource(data=dict(
        x1=dt.index[inc],
        top1=dt.Open[inc],
        bottom1=dt.Close[inc],
        high1=dt.High[inc],
        low1=dt.Low[inc],
        Date1=dt.Date[inc]
    ))

    dec_source = ColumnDataSource(data=dict(
        x2=dt.index[dec],
        top2=dt.Open[dec],
        bottom2=dt.Close[dec],
        high2=dt.High[dec],
        low2=dt.Low[dec],
        Date2=dt.Date[dec]
    ))
    # Plot candles
    # High and low
    fig.segment(x0="x1", y0="high1", x1="x1", y1="low1", source=inc_source, color=INCREASING_COLOR)
    fig.segment(x0="x2", y0="high2", x1="x2", y1="low2", source=dec_source, color=DECREASING_COLOR)

    # Open and close
    r1 = fig.vbar(x="x1", width=width, top="top1", bottom="bottom1", source=inc_source,
                  fill_color=INCREASING_COLOR, line_color="black")
    r2 = fig.vbar(x="x2", width=width, top="top2", bottom="bottom2", source=dec_source,
                  fill_color=DECREASING_COLOR, line_color="black")

    # Add on extra lines (e.g. moving averages) here

    # Add on a vertical line to indicate a trading signal here

    # Add date labels to x axis
    fig.xaxis.major_label_overrides = {
        i: date.strftime(xaxis_dt_format) for i, date in enumerate(pd.to_datetime(df["Date"]))
    }

    # Set up the hover tooltip to display some useful data
    fig.add_tools(HoverTool(
        renderers=[r1],
        tooltips=[
            ("Open", "$@top1"),
            ("High", "$@high1"),
            ("Low", "$@low1"),
            ("Close", "$@bottom1"),
            ("Date", "@Date1{" + xaxis_dt_format + "}"),
        ],
        formatters={
            "Date1": "datetime",
        }))

    fig.add_tools(HoverTool(
        renderers=[r2],
        tooltips=[
            ("Open", "$@top2"),
            ("High", "$@high2"),
            ("Low", "$@low2"),
            ("Close", "$@bottom2"),
            ("Date", "@Date2{" + xaxis_dt_format + "}")
        ],
        formatters={
            "Date2": "datetime"
        }))

    # JavaScript callback function to automatically zoom the Y axis to
    # view the data properly
    source = ColumnDataSource({"Index": dt.index, "High": dt.Max, "Low": dt.Min})
    callback = CustomJS(args={"y_range": fig.y_range, "source": source}, code="""
                                clearTimeout(window._autoscale_timeout);
                                var Index = source.data.Index,
                                    Low = source.data.Low,
                                    High = source.data.High,
                                    start = cb_obj.start,
                                    end = cb_obj.end,
                                    min = Infinity,
                                    max = -Infinity;
                                for (var i=0; i < Index.length; ++i) {
                                    if (start <= Index[i] && Index[i] <= end) {
                                        max = Math.max(High[i], max);
                                        min = Math.min(Low[i], min);
                                    }
                                }
                                var pad = (max - min) * .05;
                                window._autoscale_timeout = setTimeout(function() {
                                    y_range.start = min - pad;
                                    y_range.end = max + pad;
                                });
                            """)

    #fig.x_range.callback = callback

    export_png(fig, filename="/Users/Maksim/Desktop/patterns/" + figur + "/" + name, webdriver=web_driver)
