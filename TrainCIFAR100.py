import matplotlib.pyplot as plt
from Models import *

dt_size = 3072           # 1データのサイズ(画像のピクセル数*チャネル数)
data_split = 0.5        # 全ピクセルのうちの補助データの割合
subdt_size = int(dt_size * data_split)   # 補助データのサイズ
maindt_size = dt_size - subdt_size  # 主データのサイズ
layers_percnet = 2      # 浸透サブネットの層数
layers_intnet = 3       # 統合サブネットの層数
percnet_size = 1000      # 浸透サブネットの各層の素子数
percfeature_size = 1000  # 浸透特徴の個数
intnet_size = 1000       # 統合サブネットの各層の素子数
output_size = 100       # 出力データのサイズ
epochs_prior = 300      # 事前学習のエポック数
epochs_perc = 1000       # 浸透学習のエポック数
epochs_adj = 200        # 微調整のエポック数
batch_size = 1024        # バッチサイズ
validation_split = 1 / 6  # 評価に用いるデータの割合
test_split = 1 / 6        # テストに用いるデータの割合
verbose = 2             # 学習進捗の表示モード
decay = 0.05            # 減衰率
optimizer = Adam(lr=0.0001)      # 最適化アルゴリズム
# callbacks = [make_tensorboard(set_dir_name='log')]  # コールバック


# ニューラルネットワークの構成
percnet, network = network((maindt_size+subdt_size,),
                           percfeature_size, output_size,
                           layers_percnet, layers_intnet,
                           percnet_size, intnet_size)

percnet.compile(optimizer=optimizer, loss=mean_squared_error)
network.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])


# CIFAR100データの読み込み
dataset = CIFAR100Dataset()
x_train, y_train, x_test, y_test = dataset.get_data()
# TrainデータとTestデータを結合して，それをTrain，Validation，Testデータに分ける．
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)
x_train_main, x_train_aux, y_train, x_val, y_val, x_test, y_test = dataset.get_main_aux_data(x, y, data_split, validation_split, test_split)


# 入力データの表示
x_train = np.concatenate([x_train_main, 0*x_train_aux], axis=1)
n = 10
plt.figure()
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_train[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(x_test[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# 学習
trainer = Trainer(percnet, network, optimizer, verbose)
history_list = LossAccHistory()
history_list = trainer.train(x_train_main, x_train_aux, y_train,
                             x_val, y_val,
                             subdt_size,
                             layers_intnet,
                             batch_size,
                             epochs_prior, epochs_perc, epochs_adj,
                             decay,
                             history_list)

# 損失と精度の評価
x_train_aux *= 0
x_train = np.concatenate([x_train_main, x_train_aux], axis=1)
score_train = network.evaluate(x_train, y_train, batch_size=batch_size)
score_val = network.evaluate(x_val, y_val, batch_size=batch_size)
score_test = network.evaluate(x_test, y_test, batch_size=batch_size)
print('Train - loss:', score_train[0], '- accuracy:', score_train[1])
print('Validation - loss:', score_val[0], '- accuracy:', score_val[1])
print('Test - loss:', score_test[0], '- accuracy:', score_test[1])

# 損失と精度をグラフにプロット
plt.figure()
plt.plot(history_list.accuracy)
plt.plot(history_list.accuracy_val)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.0, 1.01)
plt.legend(['Train', 'Validation'])

plt.figure()
plt.plot(history_list.losses)
plt.plot(history_list.losses_val)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


