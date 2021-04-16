#! -*- coding: utf-8 -*-
# 2021搜狐校园文本匹配算法大赛baseline
# 直接用 RoFormer + Cond LayerNorm 融合为一个模型

import json
import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.layers import Input, Embedding, Reshape, GlobalAveragePooling1D, Dense
from keras.models import Model
from tqdm import tqdm
import jieba
jieba.initialize()

# 基本信息
maxlen = 512
epochs = 5
batch_size = 16
learing_rate = 2e-5

# bert配置
config_path = '/root/kg/bert/chinese_roformer_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer_L-12_H-768_A-12/vocab.txt'

variants = [
    u'短短匹配A类',
    u'短短匹配B类',
    u'短长匹配A类',
    u'短长匹配B类',
    u'长长匹配A类',
    u'长长匹配B类',
]

# 读取数据
train_data, valid_data, test_data = [], [], []
for i, var in enumerate(variants):
    key = 'labelA' if 'A' in var else 'labelB'
    fs = [
        '../datasets/sohu2021_open_data_clean/%s/train.txt' % var,
        '../datasets/round2/%s.txt' % var
    ]
    for f in fs:
        with open(f) as f:
            for l in f:
                l = json.loads(l)
                train_data.append((i, l['source'], l['target'], int(l[key])))
    f = '../datasets/sohu2021_open_data_clean/%s/valid.txt' % var
    with open(f) as f:
        for l in f:
            l = json.loads(l)
            valid_data.append((i, l['source'], l['target'], int(l[key])))
    f = '../datasets/sohu2021_open_data_clean/%s/test_with_id.txt' % var
    with open(f) as f:
        for l in f:
            l = json.loads(l)
            test_data.append((i, l['source'], l['target'], l['id']))

# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_conds, batch_labels = [], []
        for is_end, (cond, source, target, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                source, target, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_conds.append([cond])
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_conds = sequence_padding(batch_conds)
                batch_labels = sequence_padding(batch_labels)
                yield [
                    batch_token_ids, batch_segment_ids, batch_conds
                ], batch_labels
                batch_token_ids, batch_segment_ids = [], []
                batch_conds, batch_labels = [], []


c_in = Input(shape=(1,))
c = Embedding(len(variants), 128)(c_in)
c = Reshape((128,))(c)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    layer_norm_cond=c,
    additional_input_layers=c_in
)

output = GlobalAveragePooling1D()(model.output)
output = Dense(2, activation='softmax')(output)

model = Model(model.inputs, output)
model.summary()

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learing_rate, ema_momentum=0.9999)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    """评测函数（A、B两类分别算F1然后求平均）
    """
    total_a, right_a = 0., 0.
    total_b, right_b = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        flag = x_true[2][:, 0] % 2
        total_a += ((y_pred + y_true) * (flag == 0)).sum()
        right_a += ((y_pred * y_true) * (flag == 0)).sum()
        total_b += ((y_pred + y_true) * (flag == 1)).sum()
        right_b += ((y_pred * y_true) * (flag == 1)).sum()
    f1_a = 2.0 * right_a / total_a
    f1_b = 2.0 * right_b / total_b
    return {'f1': (f1_a + f1_b) / 2, 'f1_a': f1_a, 'f1_b': f1_b}


def predict_test(filename):
    """测试集预测到文件
    """
    with open(filename, 'w') as f:
        f.write('id,label\n')
        for x_true, y_true in tqdm(test_generator):
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            for id, y in zip(y_true, y_pred):
                f.write('%s,%s\n' % (id, y))


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        optimizer.apply_ema_weights()
        metrics = evaluate(valid_generator)
        if metrics['f1'] > self.best_val_f1:
            self.best_val_f1 = metrics['f1']
            model.save_weights('best_model.weights')
        optimizer.reset_old_weights()
        metrics['best_f1'] = self.best_val_f1
        print(metrics)


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_test('test.csv')
