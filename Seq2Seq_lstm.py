import random
import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
    multiply, concatenate, Flatten, Activation, dot
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.callbacks import EarlyStopping
import pydot as pyd
from keras.utils.vis_utils import plot_model, model_to_dot
import matplotlib
import warnings

warnings.filterwarnings('ignore')
keras.utils.vis_utils.pydot = pyd
matplotlib.use('TKAgg')

# # plot the data
# df = pd.read_csv('tesla.csv', index_col=['Date'])
# df = df.drop('Volume', axis=1)
# df.plot()
# plt.show()
# # specify columns to plot

df = pd.read_csv('tesla.csv', index_col=[0])
# detrend = signal.detrend(df)
# detrend_df = pd.DataFrame(detrend)
# detrend_df.plot(subplots=True, layout=(2, 3))
# plt.show()
# res = seasonal_decompose(df.values, model='multiplicative', extrapolate_trend='freq', period=6)
# detrend = df.values / res.trend
detrend_df = pd.DataFrame(df.values)
# detrend_df.plot(subplots=True, layout=(2, 3))
# print(detrend_df)
# plt.show()

# # # date_col = df.iloc[:, [0]]
# # # df = df.drop('Date', axis=1)
# # # print(df.head(5))

# MinMaxNormalize
values = detrend_df.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
scaled = pd.DataFrame(scaled)
# scaled.plot(subplots=True, layout=(2, 3))
# plt.show()

# Preprocess split into train and test (8:2), Since the sequence length is 640,
# the first 511 data points will be used as our train data
train_ratio = 0.8
train_len = int(train_ratio * scaled.shape[0])
print(train_len)

train_data = scaled.iloc[:train_len]
test_data = scaled.iloc[train_len:]

# Combine sequences
x_index = np.array(range(len(scaled)))
x_normalize = np.column_stack([scaled, x_index, [1] * train_len + [0] * (len(x_index) - train_len)])

# print(x_lbl)
# x_train_max = x_lbl[x_lbl[:, 7] == 1, :6].max(axis=0)
# x_train_max = x_train_max.tolist() + [1] * 6  # only normalize for the first 2 columns
# print(x_train_max)
# x_normalize = np.divide(x_lbl, x_train_max)
# print(x_normalize)


# Truncate: Next, we will cut sequence into smaller pieces by sliding an input window (length = 200 time steps) and an
# output window (length = 20 time steps), and put these samples in 3d numpy arrays.
def truncate(x, feature_cols=range(7), target_cols=range(7), label_col=7, train_len=10, test_len=2):
    in_, out_, lbl = [], [], []
    for i in range(len(x) - train_len - test_len + 1):
        in_.append(x[i:(i + train_len), feature_cols].tolist())
        out_.append(x[(i + train_len):(i + train_len + test_len), target_cols].tolist())
        lbl.append(x[i + train_len, label_col])
    return np.array(in_), np.array(out_), np.array(lbl)


X_in, X_out, lbl = truncate(x_normalize, feature_cols=range(7), target_cols=range(7),
                            label_col=7, train_len=10, test_len=2)
print(X_in.shape, X_out.shape, lbl.shape)

X_input_train = X_in[np.where(lbl == 1)]
X_output_train = X_out[np.where(lbl == 1)]
X_input_test = X_in[np.where(lbl == 0)]
X_output_test = X_out[np.where(lbl == 0)]
print(X_input_train.shape, X_output_train.shape)
print(X_input_test.shape, X_output_test.shape)
# Here we will use np.polyfit to complete this small task. Note that only the first 800 data points are used to fit
# the trend lines, this is because we want to avoid data leak.
# x1 = df.iloc[:, [0]]
# x2 = df.iloc[:, [1]]
# x3 = df.iloc[:, [2]]
# x4 = df.iloc[:, [3]]
# x5 = df.iloc[:, [4]]
# x6 = df.iloc[:, [5]]

n_hidden = 10

input_train = Input(shape=(X_input_train.shape[1], X_input_train.shape[2] - 1))
output_train = Input(shape=(X_output_train.shape[1], X_output_train.shape[2] - 1))
# print(input_train)
# print(output_train)

encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(
    n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
    return_sequences=False, return_state=True)(input_train)
# print(encoder_last_h1)
# print(encoder_last_h2)
# print(encoder_last_c)

encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
decoder = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, return_state=False,
               return_sequences=True)(
    decoder, initial_state=[encoder_last_h1, encoder_last_c])
# print(decoder)

out = TimeDistributed(Dense(output_train.shape[2]))(decoder)
# print(out)


model = Model(inputs=input_train, outputs=out)
opt = Adam(lr=0.01, clipnorm=1)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
# model.summary()


# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Training:
epc = 100
es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
history = model.fit(X_input_train[:, :, :6], X_output_train[:, :, :6], validation_split=0.2,
                    epochs=epc, verbose=1, callbacks=[es],
                    batch_size=100)
train_mae = history.history['mae']
valid_mae = history.history['val_mae']

# model.save('model_forecasting_seq2seq.h5')

# plt.plot(train_mae, label='train mae'),
# plt.plot(valid_mae, label='validation mae')
# plt.ylabel('mae')
# plt.xlabel('epoch')
# plt.title('train vs. validation accuracy (mae)')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
# plt.show()

train_pred_detrend = model.predict(X_input_train[:, :, :6])
test_pred_detrend = model.predict(X_input_test[:, :, :6])
# print(train_pred_detrend.shape, test_pred_detrend.shape)
train_true_detrend = X_output_train[:, :, :6]
test_true_detrend = X_output_test[:, :, :6]
# print(train_true_detrend.shape, test_true_detrend.shape)


train_pred_detrend = np.concatenate([train_pred_detrend, np.expand_dims(X_output_train[:, :, 6], axis=2)], axis=2)
test_pred_detrend = np.concatenate([test_pred_detrend, np.expand_dims(X_output_test[:, :, 6], axis=2)], axis=2)
print(train_pred_detrend.shape, test_pred_detrend.shape)
train_true_detrend = np.concatenate([train_true_detrend, np.expand_dims(X_output_train[:, :, 6], axis=2)], axis=2)
test_true_detrend = np.concatenate([test_true_detrend, np.expand_dims(X_output_test[:, :, 6], axis=2)], axis=2)
print(train_pred_detrend.shape, test_pred_detrend.shape)

# x1_trend_param = np.polyfit(x_index[:train_len], df.High[:train_len], 6)
# x2_trend_param = np.polyfit(x_index[:train_len], df.Low[:train_len], 5)
# x3_trend_param = np.polyfit(x_index[:train_len], df.Low[:train_len], 4)
# x4_trend_param = np.polyfit(x_index[:train_len], df.Low[:train_len], 3)
# x5_trend_param = np.polyfit(x_index[:train_len], df.Low[:train_len], 2)
# x6_trend_param = np.polyfit(x_index[:train_len], df.Low[:train_len], 1)
data_final = dict()
for dt, lb in zip([train_pred_detrend, train_true_detrend, test_pred_detrend, test_true_detrend],
                  ['train_pred', 'train_true', 'test_pred', 'test_true']):
    dt_High = dt[:, :, 0]
    # + dt[:, :, 6] * x1_trend_param[0] + dt[:, :, 6] * x1_trend_param[1] + dt[:, :, 6] * x1_trend_param[2] + dt[:, :, 6] * x1_trend_param[3] + dt[:, :, 6] * x1_trend_param[4] + dt[:, :, 5] * x1_trend_param[5] + x1_trend_param[6]
    dt_Low = dt[:, :, 1]
    # + dt[:, :, 6] * x2_trend_param[0] + dt[:, :, 6] * x2_trend_param[1] + dt[:, :, 6] * x2_trend_param[2] + dt[:, :, 6] * x2_trend_param[3] + dt[:, :, 6] * x2_trend_param[4] + x2_trend_param[5]
    dt_Open = dt[:, :, 2]
    # + dt[:, :, 6] * x3_trend_param[0] + dt[:, :, 6] * x3_trend_param[1] + dt[:, :, 6] * x3_trend_param[2] + dt[:, :, 6] * x3_trend_param[3] + x3_trend_param[4]
    dt_Close = dt[:, :, 3]
    # + dt[:, :, 6] * x4_trend_param[0] + dt[:, :, 6] * x4_trend_param[1] + dt[:, :, 6] * x4_trend_param[2] + x4_trend_param[3]
    dt_Volume = dt[:, :, 4]
    # + dt[:, :, 6] * x5_trend_param[0] + dt[:, :, 6] * x5_trend_param[1] + x5_trend_param[2]
    dt_Adj_close = dt[:, :, 5]
    # + dt[:, :, 1] * x6_trend_param[0] + x6_trend_param[1]
    data_final[lb] = np.concatenate(
        [np.expand_dims(dt_High, axis=2), np.expand_dims(dt_Low, axis=2), np.expand_dims(dt_Open, axis=2),
         np.expand_dims(dt_Close, axis=2), np.expand_dims(dt_Volume, axis=2), np.expand_dims(dt_Adj_close, axis=2)],
        axis=2)
    print(lb + ': {}'.format(data_final[lb].shape))

for lb in ['train', 'test']:
    plt.figure(figsize=(15, 4))
    plt.hist(data_final[lb + '_pred'].flatten(), bins=100, color='orange', alpha=0.5, label=lb + ' pred')
    plt.hist(data_final[lb + '_true'].flatten(), bins=100, color='green', alpha=0.5, label=lb + ' true')
    plt.legend()
    plt.title('value distribution: ' + lb)
    plt.show()


for lb in ['train', 'test']:
    MAE_overall = abs(data_final[lb+'_pred'] - data_final[lb+'_true']).mean()
    MAE_ = abs(data_final[lb+'_pred'] - data_final[lb+'_true']).mean(axis=(1, 2))
    plt.figure(figsize=(15, 3))
    plt.plot(MAE_)
    plt.title('MAE '+lb+': overall MAE = '+str(MAE_overall))
    plt.show()


ith_timestep = random.choice(range(data_final[lb + '_pred'].shape[1]))
plt.figure(figsize=(15, 5))
train_start_t = 0
test_start_t = data_final['train_pred'].shape[0]
for lb, tm, clrs in zip(['train', 'test'], [train_start_t, test_start_t], [['green', 'red'], ['blue', 'orange']]):
    for i, x_lbl in zip([0, 1, 3, 4, 5], ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']):
        plt.plot(range(tm, tm + data_final[lb + '_pred'].shape[0]),
                 data_final[lb + '_pred'][:, ith_timestep, i],
                 linestyle='solid', linewidth=1, color=clrs[0], label='pred ' + x_lbl)
        plt.plot(range(tm, tm + data_final[lb + '_pred'].shape[0]),
                 data_final[lb + '_true'][:, ith_timestep, i],
                 linestyle='--', linewidth=1, color=clrs[1], label='true ' + x_lbl)
plt.title('{}th time step in all samples'.format(ith_timestep))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=8)
plt.show()


lb = 'test'
plt.figure(figsize=(15, 5))
for i, x_lbl, clr in zip([0, 1, 2, 3, 4, 5], ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'], ['green', 'blue', 'red', 'yellow', 'brown', 'black']):
    plt.plot(data_final[lb+'_pred'][:, ith_timestep, i], linestyle='solid', color=clr, label='pred '+x_lbl)
    plt.plot(data_final[lb+'_true'][:, ith_timestep, i], linestyle='--', color=clr, label='true '+x_lbl)
plt.title('({}): {}th time step in all samples'.format(lb, ith_timestep))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
plt.show()