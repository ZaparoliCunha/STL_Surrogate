import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math as mth

import plotly.io as pio
pio.renderers.default='svg'#'browser'
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import time
import csv   

def statistical_eval(test, pred, unit, name):

    MSE=(mean_squared_error(test, pred))
    MAE=(mean_absolute_error(test, pred))
    Error_rel=(np.mean(np.mean(100*(abs(test-pred)/abs(test)))))
    if mth.isinf(Error_rel): Error_rel = 100.0
    Error_max=(np.max(np.max(np.abs(test-pred))))
    Mean_Max_Error=(np.mean(np.max(np.abs(test-pred),axis=0)))


    L = ["\n",name, "- Accuracy \n",
         " - MSE: %.2f" % MSE, unit,"\n", " - RMSE: %.2f" % MSE**(1/2.0),u"\n",
         " - MAE: %.2f" % MAE, unit, u"\n",
         " - Relative Error: ${:7.2f} %".format(Error_rel), "\n",
         " - Absolute Maximum Error: %.2f" %  Error_max, unit,"\n",
        " - Mean Absolute Maximum Error per sample: %.2f" %  Mean_Max_Error, unit,"\n",]
    print (''.join(L))
    Texto = (''.join(L).replace('\u03bc', 'u'))

    return MSE, MAE, Error_rel, Error_max, Mean_Max_Error, Texto


def plot_TruexPred(test,pred,unit, name, path, Anot):

    title = ''
    fig = px.scatter(x=test, y=pred,
                     labels={'x':'True Values'+'['+unit+']', 'y':'Predictions'+'['+unit+']'},
                     title=title + name +' - ' + Anot, width=500, height=400)
    fig.update_traces(marker={'size': 10})
    fig.update_layout(template='simple_white')
    fig.add_shape(type="line", line_color="salmon", line_width=5, opacity=1, line_dash="dot",
        x0=-1e6, x1=1e10, xref="paper", y0=-1e6, y1=1e10, yref="y")
    fig.update_layout(font_size=14, title_font_size=18, title_xref ='paper', title_x = 0.5, title_y = 0.9)
    fig.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}])
    fig.show()
    fig.write_image('./'+path+'/'+  name+".pdf")
    return


def plot_ErrorByFreq(freq,erro, path):       
             
    file_ErrorByFreq_ = open('./'+path+'/'+ 'ErrorByFreq.csv','w')
    writer_ErrorByFreq_ = csv.writer(file_ErrorByFreq_)
    writer_ErrorByFreq_.writerow(list(erro))
    file_ErrorByFreq_.close()    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=erro, mode='lines+markers',line=dict(width=4)))
    fig.update_xaxes(type="log")
    fig.update_layout(title= 'Error per Frequency',
                xaxis_title="Frequency [Hz]",
                yaxis_title = 'RMSE [dB]',
                template='simple_white',
                font_size=14, title_font_size=18, title_xref ='paper', title_x = 0.5, title_y = 0.9,
                width=500, height=400)
    fig.show()
    fig.write_image('./'+path+'/'+ "ErrorByFreq.pdf")    
    return

def evaluate_ML(path, title, model, X_test, y_test, outputs, units,
                   N_samples, start_time, report, writer, freq,
                   preprocess_pipe = None, output_transf = None, curve = False, curve_var = ''):
    
    print(''.join(["\n ", "Elapsed Time to Fit: %.2f" %  (time.time() - start_time), 's',"\n"]))    
    report.write(''.join(["#### \n ", "Elapsed Time to Fit: %.2f" %  (time.time() - start_time), 's',"\n", "#### \n "]))

    out_shape = y_test.shape[1]
    X_t = X_test.copy()
    if preprocess_pipe is not None:
        X_t = preprocess_pipe.transform(X_t)
    ypred = model.predict(X_t).reshape(-1,out_shape)
    if output_transf is not None:
        if ypred.ndim == 1: ypred = ypred.reshape(-1,1)
        if y_test.ndim == 1: y_test = y_test.values.reshape(-1,1)
        ypred = output_transf.inverse_transform(ypred).reshape(-1,out_shape)

    if curve:

        MSE, MAE, Error_rel, Error_max, Mean_Max_Error,Texto = \
            statistical_eval(y_test, ypred, units, curve_var)
        report.write(Texto)        
        Anot = "RMSE: %.2f"% (MSE**(1/2.0)) + units
        plot_TruexPred(y_test.stack(), pd.DataFrame(ypred).stack(),units, curve_var, path, Anot)


        sample_ind= 1
        X_comp = X_test.iloc[[sample_ind]].copy(deep=True)
        Y_comp = y_test.iloc[[sample_ind]].copy(deep=True)
        if preprocess_pipe is not None:
            X_comp = preprocess_pipe.transform(X_comp)               
        
        #Depending on the tensorflow version use line 107 to replace line 106
        ycomp = model.predict(X_comp).reshape(-1,out_shape)
        #ycomp = model.predict(pd.concat([X_comp, X_comp], axis= 0))[0,:].reshape(-1,out_shape)
        if output_transf is not None:
            ycomp = output_transf.inverse_transform(ycomp).reshape(-1,out_shape)           

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freq, y=np.asarray(Y_comp)[0], name='Simulated',
                                  mode='lines',line=dict(width=4)))
        fig.add_trace(go.Scatter(x=freq, y=ycomp[0], name='Predicted',
                                  line=dict(color="salmon", width=4, dash='dash')))
        fig.update_xaxes(type="log")
        fig.update_layout(title= title +' - ' + Anot,
                    xaxis_title="Frequency [Hz]",
                    yaxis_title = curve_var + ' ['+units+']',
                    template='simple_white',
                    font_size=14, title_font_size=18, title_xref ='paper', title_x = 0.5, title_y = 0.9,
                    width=500, height=400,
                    )
        fig.show()
        fig.write_image('./'+path+'/'+ "Sample_Curve.pdf")
        
        plot_error_by_freq = True
        if plot_error_by_freq: plot_ErrorByFreq(freq, np.sqrt(np.mean(np.square(y_test.values-ypred),axis=0)), path)

    else:
        MSE, MAE, Error_rel, Error_max, Mean_Max_Error = [],[],[],[],[]
        print("\n\n\n"+title+":\n")
        report.write("\n\n\n"+title+":\n")
        for i, out in enumerate (outputs):
            iMSE, iMAE, iError_rel, iError_max, iMME,Texto = \
                statistical_eval(y_test.iloc[:,i], ypred[:,i], units[i], out)
            Anot = "RMSE: %.2f"% (iMSE**(1/2.0)) + units[i]
            plot_TruexPred(y_test.iloc[:,i], ypred[:,i],units[i], out, path, Anot)
            MSE.append(iMSE)
            MAE.append(iMAE)
            Error_rel.append(iError_rel)
            Error_max.append(iError_max)
            Mean_Max_Error.append(iMME)
            report.write(Texto)
    writer.writerow([N_samples, Error_rel, MSE, MAE, Error_max, Mean_Max_Error,(time.time() - start_time)])
      
    return


def plot_hist(results_path, hist):
    corte = 200; 
    train_mae = hist.history['loss'][corte:]
    epochs = range(corte , len(train_mae) + corte )
    plt.figure( dpi=160)
    plt.plot(epochs,train_mae,linewidth=2)
    plt.plot(epochs,hist.history['val_loss'][corte:],'-.',linewidth=1)
    plt.title('Loss (Weighted MSE)')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training','validation'], loc = 'upper right')
    plt.savefig('./'+results_path+'/'+'Hist_convergence.png')
    plt.show()
