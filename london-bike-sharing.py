# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:00.166866Z","iopub.execute_input":"2023-07-24T02:48:00.167524Z","iopub.status.idle":"2023-07-24T02:48:11.559692Z","shell.execute_reply.started":"2023-07-24T02:48:00.167488Z","shell.execute_reply":"2023-07-24T02:48:11.558274Z"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import datetime
import sys

print('python version :',sys.version)
print('tensorflow version :',tf.__version__)

np.random.seed(123)
tf.keras.utils.set_random_seed(456)

# %% [markdown]
# # Data Wrangling & Exploration

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.563451Z","iopub.execute_input":"2023-07-24T02:48:11.564421Z","iopub.status.idle":"2023-07-24T02:48:11.657346Z","shell.execute_reply.started":"2023-07-24T02:48:11.564372Z","shell.execute_reply":"2023-07-24T02:48:11.656079Z"}}
bike_sharing = pd.read_csv('/kaggle/input/london-bike-sharing-dataset/london_merged.csv')
bike_sharing.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.658739Z","iopub.execute_input":"2023-07-24T02:48:11.659075Z","iopub.status.idle":"2023-07-24T02:48:11.694992Z","shell.execute_reply.started":"2023-07-24T02:48:11.659045Z","shell.execute_reply":"2023-07-24T02:48:11.693704Z"}}
bike_sharing.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.696491Z","iopub.execute_input":"2023-07-24T02:48:11.696857Z","iopub.status.idle":"2023-07-24T02:48:11.704568Z","shell.execute_reply.started":"2023-07-24T02:48:11.696827Z","shell.execute_reply":"2023-07-24T02:48:11.703376Z"}}
# helper function 

def hours_count(df,sort_by):
    day_hour_df = pd.DataFrame()
    day_hour_df['daydate'] = df.timestamp.dt.date
    day_hour_df['hour'] = df.timestamp.dt.hour
    hours_count = day_hour_df.groupby(by='daydate').agg({'hour':'count'}).reset_index()
    hours_count = hours_count.sort_values(by=sort_by)
    hours_count['missing'] = 24 - hours_count.hour
    print('total missing =',hours_count['missing'].sum())
    return hours_count

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.708181Z","iopub.execute_input":"2023-07-24T02:48:11.708616Z","iopub.status.idle":"2023-07-24T02:48:11.762510Z","shell.execute_reply.started":"2023-07-24T02:48:11.708585Z","shell.execute_reply":"2023-07-24T02:48:11.761450Z"}}
bike_sharing.timestamp = pd.to_datetime(bike_sharing.timestamp)
hours_count(bike_sharing, sort_by='hour').head()

# %% [markdown]
# look like we still have missing data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.763913Z","iopub.execute_input":"2023-07-24T02:48:11.764388Z","iopub.status.idle":"2023-07-24T02:48:11.824479Z","shell.execute_reply.started":"2023-07-24T02:48:11.764346Z","shell.execute_reply":"2023-07-24T02:48:11.823331Z"}}
# adding missing hour
min_datetime = bike_sharing.timestamp.min()
max_datetime = bike_sharing.timestamp.max()
all_datetime = pd.date_range(min_datetime, max_datetime, freq='H')
datetime_df = pd.DataFrame({'timestamp':all_datetime})
new_bike_sharing = pd.merge(datetime_df, bike_sharing, on='timestamp', how='left')

hours_count(new_bike_sharing, sort_by='hour').head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.827263Z","iopub.execute_input":"2023-07-24T02:48:11.827721Z","iopub.status.idle":"2023-07-24T02:48:11.838270Z","shell.execute_reply.started":"2023-07-24T02:48:11.827690Z","shell.execute_reply":"2023-07-24T02:48:11.837091Z"}}
print(f'==== missing values ==== \n{new_bike_sharing.isna().sum()}')
print(f'==== duplicated values ==== \n {new_bike_sharing.timestamp.duplicated().value_counts()}')                          
# new_bike_sharing.sample(10)

# %% [markdown]
# The missing values for each col should be 106 but the result of `new_bike_sharing.isna().sum()` is 130 each. Probably because there is a totaly missing value in a one day (24 hours)

# %% [code] {"_kg_hide-input":false,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.840008Z","iopub.execute_input":"2023-07-24T02:48:11.840440Z","iopub.status.idle":"2023-07-24T02:48:11.887369Z","shell.execute_reply.started":"2023-07-24T02:48:11.840396Z","shell.execute_reply":"2023-07-24T02:48:11.886209Z"}}
null_count_df = pd.DataFrame()
null_count_df['daydate'] = new_bike_sharing.timestamp.dt.date
null_count_df['nulls_sum'] = new_bike_sharing.isna().astype(int).sum(axis=1)
null_count_df['missing_hours'] = new_bike_sharing.isna().astype(int).cnt
null_count_df = null_count_df.groupby('daydate').agg('sum').reset_index()
null_count_df.sort_values('nulls_sum', ascending=False).reset_index(drop=True)

# %% [markdown]
# i am right,basically no data recorded for 2 September 2016

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.889159Z","iopub.execute_input":"2023-07-24T02:48:11.889978Z","iopub.status.idle":"2023-07-24T02:48:11.953305Z","shell.execute_reply.started":"2023-07-24T02:48:11.889935Z","shell.execute_reply":"2023-07-24T02:48:11.952268Z"}}
Sep9 = new_bike_sharing.timestamp.dt.date.astype(str) == '2016-09-02'
# 9 September 2016 is Friday and not a Holiday
new_bike_sharing.loc[Sep9,['is_holiday','is_weekend']] = 0

#filling other missing values
new_bike_sharing.is_holiday.interpolate('pad', inplace=True)
new_bike_sharing.is_weekend.interpolate('pad', inplace=True)
new_bike_sharing.season.interpolate('pad', inplace=True)
values = {
    'cnt':new_bike_sharing.cnt.mean(),
    't1':new_bike_sharing.t1.mean(),
    't2':new_bike_sharing.t2.mean(),
    'hum':new_bike_sharing.hum.mean(),
    'wind_speed':new_bike_sharing.wind_speed.mean(),
    'weather_code':new_bike_sharing.weather_code.mean(),
}
new_bike_sharing.fillna(values,inplace=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.954848Z","iopub.execute_input":"2023-07-24T02:48:11.955557Z","iopub.status.idle":"2023-07-24T02:48:11.978268Z","shell.execute_reply.started":"2023-07-24T02:48:11.955514Z","shell.execute_reply":"2023-07-24T02:48:11.977125Z"}}
new_bike_sharing.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:11.979513Z","iopub.execute_input":"2023-07-24T02:48:11.980308Z","iopub.status.idle":"2023-07-24T02:48:12.352833Z","shell.execute_reply.started":"2023-07-24T02:48:11.980254Z","shell.execute_reply":"2023-07-24T02:48:12.351540Z"}}
bike_sharing.iloc[:,2:].corrwith(bike_sharing.cnt).plot\
.barh(title='correlation with cnt (old df)',)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T02:48:12.354549Z","iopub.execute_input":"2023-07-24T02:48:12.354990Z","iopub.status.idle":"2023-07-24T02:48:12.680735Z","shell.execute_reply.started":"2023-07-24T02:48:12.354949Z","shell.execute_reply":"2023-07-24T02:48:12.679584Z"}}
new_bike_sharing.iloc[:,2:].corrwith(new_bike_sharing.cnt).plot\
.barh(title='correlation with cnt (new df)',)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:12.681983Z","iopub.execute_input":"2023-07-24T02:48:12.682329Z","iopub.status.idle":"2023-07-24T02:48:12.690772Z","shell.execute_reply.started":"2023-07-24T02:48:12.682299Z","shell.execute_reply":"2023-07-24T02:48:12.689687Z"}}
def plot_series(time, series, format="-", start=0, end=None, title='',legend=None):

    plt.figure(figsize=(10, 6))
    
    if type(series) is tuple:
        for series_num in series:
            plt.plot(time[start:end], series_num[start:end], format)
        plt.legend(legend)
    else:
        plt.plot(time[start:end], series[start:end], format)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:12.697852Z","iopub.execute_input":"2023-07-24T02:48:12.698269Z","iopub.status.idle":"2023-07-24T02:48:13.243972Z","shell.execute_reply.started":"2023-07-24T02:48:12.698234Z","shell.execute_reply":"2023-07-24T02:48:13.242737Z"}}
plot_series(new_bike_sharing.timestamp, new_bike_sharing.cnt, title='bike share count plot')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T02:48:13.245555Z","iopub.execute_input":"2023-07-24T02:48:13.245922Z","iopub.status.idle":"2023-07-24T02:48:15.719468Z","shell.execute_reply.started":"2023-07-24T02:48:13.245888Z","shell.execute_reply":"2023-07-24T02:48:15.717560Z"}}
plt.figure(figsize=(10, 6))
ax = sns.pointplot(x=new_bike_sharing.timestamp.dt.hour, y='cnt',hue='is_weekend',data=new_bike_sharing)
ax.set_title("comparison daily bike share between weekend and non weekend")
ax.set_xlabel("hour")
ax.set_ylabel('cnt')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T02:48:15.720988Z","iopub.execute_input":"2023-07-24T02:48:15.721618Z","iopub.status.idle":"2023-07-24T02:48:18.129901Z","shell.execute_reply.started":"2023-07-24T02:48:15.721582Z","shell.execute_reply":"2023-07-24T02:48:18.128761Z"}}
plt.figure(figsize=(10, 6))
ax = sns.pointplot(x=new_bike_sharing.timestamp.dt.hour, y='cnt',hue='is_holiday',data=new_bike_sharing)
ax.set_title("comparison daily bike share between holiday and non holiday")
ax.set_xlabel("hour")
ax.set_ylabel('cnt')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T02:48:18.131370Z","iopub.execute_input":"2023-07-24T02:48:18.131924Z","iopub.status.idle":"2023-07-24T02:48:19.062842Z","shell.execute_reply.started":"2023-07-24T02:48:18.131892Z","shell.execute_reply":"2023-07-24T02:48:19.061980Z"}}
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=new_bike_sharing.timestamp.dt.year, y='cnt',hue='season',data=new_bike_sharing)
ax.set_title("comparison between seasons {0-spring ; 1-summer; 2-fall; 3-winter.}")
ax.set_xlabel("year")
ax.set_ylabel('cnt')
plt.show()

# %% [markdown]
# # Preparing Data

# %% [markdown]
# split the data for training and testing

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:19.063970Z","iopub.execute_input":"2023-07-24T02:48:19.064544Z","iopub.status.idle":"2023-07-24T02:48:19.071553Z","shell.execute_reply.started":"2023-07-24T02:48:19.064507Z","shell.execute_reply":"2023-07-24T02:48:19.070334Z"}}
split = int(len(new_bike_sharing) * .8)
train_df = new_bike_sharing[:split]
test_df = new_bike_sharing[split:]

# %% [markdown]
# creating object for series datasets

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:19.073197Z","iopub.execute_input":"2023-07-24T02:48:19.073582Z","iopub.status.idle":"2023-07-24T02:48:19.088131Z","shell.execute_reply.started":"2023-07-24T02:48:19.073550Z","shell.execute_reply":"2023-07-24T02:48:19.086862Z"}}
class Data:pass
class SeriesDataset:
    def __init__(self,train_df,test_df,features=[], window_size = 24, batch_size=32):
        self.window_size = window_size 
        self.batch_size =  batch_size
        kwargs = {
            'features': features,
            'label':'cnt',
            'window_size' : self.window_size,
            'batch_size' : self.batch_size,
        }
       
        self.train_dataset = self.windowed_dataset(train_df,shuffle_buffer=10000,**kwargs)
        self.test_dataset = self.windowed_dataset(test_df,**kwargs)
    
    def windowed_dataset(self, df, 
                         features, 
                         label=None,  
                         window_size=30, 
                         shift=1, 
                         batch_size=32, 
                         shuffle_buffer=None):
        
        if label is not None:
            dataset_x = tf.data.Dataset.from_tensor_slices(df[features])
            dataset_y = tf.data.Dataset.from_tensor_slices(df[label])
            dataset_x = dataset_x.window(window_size +1 , shift=1, drop_remainder=True)
            dataset_x = dataset_x.flat_map(lambda window: window.batch(window_size+1))
            dataset_x = dataset_x.map(lambda window: window[:-1])
            dataset =  tf.data.Dataset.zip((dataset_x,dataset_y))
        else :
            dataset = tf.data.Dataset.from_tensor_slices(df[features])
            dataset = dataset.window(window_size + 1 , shift=1, drop_remainder=True)
            dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
            dataset = dataset.map(lambda window: (window[:-1],window[-1]))
        
        if shuffle_buffer is not None:
            dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size).prefetch(1)
    
        return dataset

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:19.089317Z","iopub.execute_input":"2023-07-24T02:48:19.089640Z","iopub.status.idle":"2023-07-24T02:48:19.484104Z","shell.execute_reply.started":"2023-07-24T02:48:19.089612Z","shell.execute_reply":"2023-07-24T02:48:19.483088Z"}}
features = ['t1', 't2', 'hum']
n_features = len(features)
series = SeriesDataset(train_df, test_df,features, batch_size=64)
#dataset for second model
cnt_train = series.windowed_dataset(train_df,'cnt',window_size=24, batch_size=64)
cnt_test = series.windowed_dataset(test_df,'cnt',window_size=24, batch_size=64)

# %% [markdown]
# # Modelling

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:19.485901Z","iopub.execute_input":"2023-07-24T02:48:19.486273Z","iopub.status.idle":"2023-07-24T02:48:20.095576Z","shell.execute_reply.started":"2023-07-24T02:48:19.486239Z","shell.execute_reply":"2023-07-24T02:48:20.094476Z"}}
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense,Normalization, BatchNormalization

input_1 = Input(shape=(series.window_size,n_features), name='windowed_features_input')
input_2 = Input(shape=(series.window_size,1), name='windowed_series_input')

norm = Normalization(name='norm')(input_1)

bidirectional_1 = Bidirectional(LSTM(64, activation='relu',name='LSTM1'),name='Bidirectional1')(norm)
bidirectional_2 = Bidirectional(LSTM(64, activation='relu',name='LSTM2'),name='Bidirectional2')(input_2)

x1 = Dense(512,activation='relu',name='dense_1.1')(bidirectional_1)
x1 = Dense(512,activation='relu',name='dense_1.2')(x1)

x2 = Dense(512,activation='relu',name='dense_2.1')(bidirectional_2)
x2 = Dense(512,activation='relu',name='dense_2.2')(x2)

output_1 = Dense(1, activation='linear',name='prediction1')(x1)
output_2 = Dense(1, activation='linear',name='prediction2')(x2)

model1 = tf.keras.Model(inputs=[input_1], outputs=[output_1], name='model_1')
model2 = tf.keras.Model(inputs=[input_2], outputs=[output_2], name='model_2')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T02:48:20.097107Z","iopub.execute_input":"2023-07-24T02:48:20.097552Z","iopub.status.idle":"2023-07-24T02:48:20.431907Z","shell.execute_reply.started":"2023-07-24T02:48:20.097508Z","shell.execute_reply":"2023-07-24T02:48:20.430655Z"}}
model1.summary()
tf.keras.utils.plot_model(model1, show_shapes=True, show_layer_names=True,show_layer_activations=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:20.433281Z","iopub.execute_input":"2023-07-24T02:48:20.433633Z","iopub.status.idle":"2023-07-24T02:48:20.534409Z","shell.execute_reply.started":"2023-07-24T02:48:20.433606Z","shell.execute_reply":"2023-07-24T02:48:20.533411Z"}}
model2.summary()
tf.keras.utils.plot_model(model2, show_shapes=True, show_layer_names=True,show_layer_activations=True)

# %% [markdown]
# # Training

# %% [markdown]
# defining callbacks

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:20.535713Z","iopub.execute_input":"2023-07-24T02:48:20.536091Z","iopub.status.idle":"2023-07-24T02:48:20.543129Z","shell.execute_reply.started":"2023-07-24T02:48:20.536058Z","shell.execute_reply":"2023-07-24T02:48:20.541803Z"}}
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0,
    patience=30,
    verbose=2,
    restore_best_weights=True,
)

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=5,
        min_lr=1e-8,
        verbose=0,
        min_delta=0,
    )

callbacks = [early_stop,lr_reduce]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T02:48:20.544604Z","iopub.execute_input":"2023-07-24T02:48:20.544945Z","iopub.status.idle":"2023-07-24T03:10:31.249603Z","shell.execute_reply.started":"2023-07-24T02:48:20.544903Z","shell.execute_reply":"2023-07-24T03:10:31.248438Z"}}
model1.compile(loss='huber', optimizer='adam',metrics=['mse'])
model1.fit(series.train_dataset,
           callbacks=callbacks,
           epochs=150,
           validation_data=series.test_dataset
          )

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T03:10:31.250973Z","iopub.execute_input":"2023-07-24T03:10:31.251339Z","iopub.status.idle":"2023-07-24T03:31:41.023657Z","shell.execute_reply.started":"2023-07-24T03:10:31.251309Z","shell.execute_reply":"2023-07-24T03:31:41.022755Z"}}
model2.compile(loss='huber', optimizer='adam',metrics=['mse'])
model2.fit(cnt_train,
           callbacks=callbacks,
           epochs=150,
           validation_data=cnt_test
          )

# %% [markdown]
# # Evaluation using test data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T03:31:41.025130Z","iopub.execute_input":"2023-07-24T03:31:41.026883Z","iopub.status.idle":"2023-07-24T03:31:44.005313Z","shell.execute_reply.started":"2023-07-24T03:31:41.026839Z","shell.execute_reply":"2023-07-24T03:31:44.004211Z"}}
pred1 = model1.predict(series.test_dataset)
loss, mse = model1.evaluate(series.test_dataset)
rmse1 = np.sqrt(mse)
plot_series(test_df.iloc[:-series.window_size].timestamp, (test_df.iloc[:-series.window_size].cnt, pred1), 
            legend=['true_value','pred1'],
            title=f'prediction1 plot, rmse={rmse1}')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T03:31:44.006503Z","iopub.execute_input":"2023-07-24T03:31:44.006805Z","iopub.status.idle":"2023-07-24T03:31:46.391619Z","shell.execute_reply.started":"2023-07-24T03:31:44.006777Z","shell.execute_reply":"2023-07-24T03:31:46.390457Z"}}
pred2 = model2.predict(cnt_test)
loss, mse = model2.evaluate(cnt_test)
rmse2 = np.sqrt(mse)
plot_series(test_df.iloc[:-series.window_size].timestamp, (test_df.iloc[:-series.window_size].cnt, pred2), 
            legend=['true_value','pred2'],
            title=f'prediction2 plot, rmse={rmse2}')

# %% [markdown]
# we will only use model2 next because it perform much better

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T03:31:46.393430Z","iopub.execute_input":"2023-07-24T03:31:46.393817Z","iopub.status.idle":"2023-07-24T03:31:46.404210Z","shell.execute_reply.started":"2023-07-24T03:31:46.393783Z","shell.execute_reply":"2023-07-24T03:31:46.402805Z"}}
#function for zooming prediction 2 plot

def zoom_plot(zoom_value):
    start = int((len(test_df) - series.window_size) * zoom_value)

    is_weekend = test_df.iloc[:-series.window_size].is_weekend * 2000
    plot_series(test_df.iloc[:-series.window_size].timestamp, 
                (test_df.iloc[:-series.window_size].cnt, pred2, is_weekend), 
                legend=['true_value','pred2','is_weekend'],
                title=f'prediction2 plot, rmse={rmse2}',
                start=start,
                end=start + 24*7
               )

# %% [markdown]
# zooming with slider widget

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T03:31:46.405543Z","iopub.execute_input":"2023-07-24T03:31:46.405888Z","iopub.status.idle":"2023-07-24T03:31:46.875505Z","shell.execute_reply.started":"2023-07-24T03:31:46.405859Z","shell.execute_reply":"2023-07-24T03:31:46.874263Z"}}
import ipywidgets as wg

slider = wg.FloatSlider(value=0, min=0, max=1,step=0.01, description='zoom ')

wg.interact(zoom_plot, zoom_value=slider)

# %% [markdown]
# # Forecasting

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-24T03:31:46.877355Z","iopub.execute_input":"2023-07-24T03:31:46.877706Z","iopub.status.idle":"2023-07-24T03:31:59.220089Z","shell.execute_reply.started":"2023-07-24T03:31:46.877676Z","shell.execute_reply":"2023-07-24T03:31:59.218878Z"}}
from ipywidgets import IntProgress
from IPython.display import display


init_x = list(new_bike_sharing.iloc[-24:].cnt.values)
predict_range = 24 * 7 #predict next week
predictions = []

#progress bar
progress = IntProgress(min=0, max=predict_range) 
display(progress) 

for i in range(predict_range):
    pred = model2.predict(np.array(init_x[-24:]).reshape(24,1), verbose=0)
    pred = list(pred[0])
    init_x.extend(pred)
    predictions.extend(pred)
    progress.value += 1

plot_series(list(range(predict_range)), 
                predictions, 
                title='Forecast result',
               )

def zoom_plot(zoom_value):
    start = int(24 * zoom_value)
    plot_series(list(range(predict_range)), 
                predictions, 
                title='Zoomed Forecast result',
                start=start,
                end=start + 24
               )
slider = wg.FloatSlider(value=0, min=0, max=6,step=1.0, description='zoom ')
wg.interact(zoom_plot, zoom_value=slider)