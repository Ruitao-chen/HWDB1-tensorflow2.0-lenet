#导入模块

import tensorflow as tf
import numpy as np
from HWDB1 import *


(x_train,y_train)=next(iter(get_HWDBdataset('train')))
(x_test,y_test)=next(iter(get_HWDBdataset('test')))



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='same',
                 input_shape=(28,28,1),  activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=36, kernel_size=(5,5), padding='same', 
    			 activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='softmax')
])

#打印模型
print(model.summary())
#训练配置
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy']) 
#开始训练
model.fit(x=x_train, y=y_train, validation_split=0.2, 
                        epochs=20, batch_size=300, verbose=2) 
