import tensorflow as tf
import pandas as pd
import numpy as np
import os
from Model_n_train import VariationalEncoder
# Generatorモデルのパス
generator_model_path = 'Generator'

# 単語リストを読み込む


# 単語のプロトタイプを読み込む関数
def load_word_prototype(word):
    
    return tf.random.normal((1, 128, 3))

# モデルをロード
generator_loaded = tf.keras.models.load_model(generator_model_path)

# シミュレーションされたジェスチャーの辞書を初期化
simulated_gestures = {}

# 単語ごとにジェスチャをシミュレート
for word in test_words_list:
    # プロトタイプを読み込む
    word_prototype = load_word_prototype(word)
    
    # 潜在変数zをサンプリング
    # エンコーダの代わりに、ノイズを生成して使用する
    # 注意: これは仮定の実装であり、実際の潜在変数はエンコーダから得られる必要があります
    z = np.random.normal(size=(1, 32))
    
    # ジェネレータを使用してジェスチャを生成
    simulated_gesture = generator_loaded.predict(tf.expand_dims(word_prototype, 0))
    simulated_gestures[word] = simulated_gesture

# 結果を表示
for word, gesture in simulated_gestures.items():
    print(f"{word}: {gesture}")