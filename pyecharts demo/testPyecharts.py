import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.font_manager as fm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pyecharts import Line, Grid, Overlap, EffectScatter, Scatter

# 正常显示中文
font = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')

def test_stationary(timeseries):
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    print('Result of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def decompose(ts):
    decomposition = seasonal_decompose(ts)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    snr = np.mean(np.abs(seasonal)) / np.mean(np.abs(residual))
    print('snr = ', snr)
    ymax = np.max([seasonal.max(), residual.max()]) * 2.0
    ymin = np.min([seasonal.min(), residual.min()]) * 2.0
    

    line1 = Line(title = '{}'.format('demo'), 
        subtitle = 'Seasonal Index = {:.2f}'.format(snr),
        subtitle_text_size = 14, title_pos = 'center')
    inx1 = ts.index.strftime("%Y-%m")
    line1.add('Original', inx1, ts.values, xaxis_interval = 1, 
              xaxis_rotate = 30, line_width = 3,
#              line_color = '#C23531', 
              line_opacity = 0.7)
    line1.add('Trend', inx1, np.round(trend.values),
            xaxis_interval = 1, xaxis_rotate = 30,
            xaxis_label_textsize = 10, yaxis_label_textsize = 10,
            legend_pos = '73%', line_opacity = 0.6,
#            '#C23531','#2F4554', #61A0A8, #D48265, #91C7AE, #812321
            label_color = ['#D48265', '#91C7AE', '#C23531', '#2F4554', '#61A0A8'],
            line_width = 3,
            is_toolbox_show = False, is_xaxislabel_align = True)
    
    sc1 = Scatter()
    sc1.add("Original", inx1, ts.values, symbol_size = 5)
    sc1.add("Trend", inx1, np.round(trend.values), symbol_size = 5)
#             label_color = ['#C23531','#2F4554'])
    overlap1 = Overlap()
    overlap1.add(line1)
    overlap1.add(sc1)

    line2 = Line()
    v = np.round(seasonal[1:13].values)
    idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
           'Oct', 'Nov', 'Dec']
    min_index = np.argmin(v)
    max_index = np.argmax(v)
    line2.add("Seasonality", idx, v,
             xaxis_interval = 0, xaxis_rotate = 30,
             yaxis_min = ymin, yaxis_max = ymax, 
             xaxis_label_textsize = 12,
             yaxis_label_textsize = 10,
             legend_top="53%", legend_pos = "20%",
             is_xaxislabel_align = True, label_color = ['#fff'],
             line_width = 3, line_opacity = 0.7, yaxis_label_textcolor = '#fff')
    
    es = EffectScatter()
    es.add("", [idx[min_index], idx[max_index]], 
           [v[min_index], v[max_index]],
           effect_scale = 3.5, effect_period = 3)
    sc2 = Scatter()
    sc2.add("Seasonality", idx, v, symbol_size = 5)
    overlap2 = Overlap()
    overlap2.add(line2)
    overlap2.add(sc2)
    overlap2.add(es)

    line3 = Line()
    residual.dropna(inplace=True)
    idx3 = residual.index.strftime("%Y-%m")
    line3.add("Residuals", idx3, np.round(residual.values), 
            yaxis_min = ymin, yaxis_max = ymax,
            xaxis_label_textsize = 10, yaxis_label_textsize = 10,
            legend_top="53%", legend_pos = "70%",
            is_toolbox_show = False, is_xaxislabel_align = True,
            line_width = 3, line_opacity = 0.7, yaxis_label_textcolor = '#fff')
    sc3 = Scatter()
    sc3.add("Residuals", idx3, np.round(residual.values), symbol_size = 5)
    overlap3 = Overlap()
    overlap3.add(line3)
    overlap3.add(sc3)
    
    grid = Grid(width=1000, height=600)

    grid.add(overlap1, grid_bottom="60%")
    grid.add(overlap2, grid_top="60%", grid_right="56%")
    grid.add(overlap3, grid_top="60%", grid_left="56%")
    grid.render(path = 'demo.html')

    # plt.figure(figsize=(20,15))
    # plt.subplot(211)
    # plt.plot(ts, label='Original')
    # plt.plot(trend, color='red', label='Trend')
    # plt.legend(loc='best')
    # plt.title(u'{}\n;Seasonal Index = {:.2f}'
    #           .format('demo'), 
    #             fontproperties=font, fontsize = 28)
   
    # plt.subplot(223)
    # s = seasonal[1:13]
    # s.index = range(1, 13)
    # min_index = np.argmin(s)
    # max_index = np.argmax(s)
    # plt.plot(s, label='Seasonality')
    # plt.xlim((1,12))
    # plt.ylim((ymin, ymax))
    # plt.annotate('peak', xy = (max_index, s[max_index]))
    # plt.annotate('vally', xy = (min_index, s[min_index]))
    # plt.plot(max_index,s[max_index],'rs')
    # plt.plot(min_index,s[min_index],'gs')
    # plt.legend(loc='best')

    # plt.subplot(224)
    # plt.plot(residual, label='Residuals')
    # plt.legend(loc='best')
    # plt.ylim((ymin, ymax))
    # plt.savefig('%s.png' % ('demo'))
    #plt.show()
    
    

citydata = pd.read_table("city_demo.txt", sep = ',')
citydata['date'] = pd.to_datetime(citydata.date)
# 用作日期对齐用
datelist = pd.read_table("datelist.txt")
datelist.d = pd.to_datetime(datelist.d)

city = citydata.city.iloc[0]
data = citydata[['date', 'data']]
data = data.merge(datelist, left_on='date', right_on='d', how='right')
data = data.fillna(0)
ts = data.set_index('d')['data']
ts.sort_index(inplace=True)
decompose(ts)


    
