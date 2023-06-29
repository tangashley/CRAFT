import tensorflow as tf
import keras
import keras.backend as K
from focal_loss import BinaryFocalLoss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, \
    Bidirectional, \
    GRU
from tensorflow.keras import initializers, regularizers, optimizers
import tensorflow_addons as tfa
from .layers import AttentionWithContext, Addition


##############################################
# LSTM AND BLSTM MODELS
##############################################

def lstm(vocab, hidden_units, num_layers, max_sequence_length, is_attention=True, is_bidirectional=True):
    timesteps = max_sequence_length
    num_classes = 2

    # adam = tf.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    adam = tf.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

    model = Sequential()
    model.add(Embedding(len(vocab), 100, input_length=max_sequence_length))

    for i in range(num_layers):
        return_sequences = is_attention or (num_layers > 1 and i < num_layers - 1)

        if is_bidirectional:
            model.add(Bidirectional(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                                         kernel_initializer=initializers.glorot_normal(seed=777),
                                         bias_initializer='zeros')))
        else:
            model.add(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                           kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros'))

        if is_attention:
            model.add(AttentionWithContext())
            model.add(Addition())

    model.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_normal(seed=777),
                    bias_initializer='zeros'))
    # model.compile(loss='categorical_crossentropy', optimizer=adam,
    #               metrics=["accuracy",
    #                        tfa.metrics.FBetaScore(num_classes=2, average="weighted", name="weighted_f1"),
    #                        keras.metrics.Precision(name='precision'),
    #                        keras.metrics.Recall(name='recall'),
    #                        keras.metrics.AUC(name='auc')])

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=adam,
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.Precision(name='precision'),
                           keras.metrics.Recall(name='recall'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.AUC(name='prc', curve='PR')])  # precision-recall curve

    model.summary()

    return model


def lstm_word_embedding(vocab, hidden_units, num_layers, max_sequence_length, is_attention, is_bidirectional,
                        word_embedding):
    timesteps = max_sequence_length
    num_classes = 2
    embedding_dim = word_embedding.shape[1]

    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

    model = Sequential()
    model.add(Embedding(len(vocab), embedding_dim, input_length=max_sequence_length, weights=[word_embedding]))

    for i in range(num_layers):
        return_sequences = is_attention or (num_layers > 1 and i < num_layers - 1)

        if is_bidirectional:
            model.add(Bidirectional(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                                         kernel_initializer=initializers.glorot_normal(seed=777),
                                         bias_initializer='zeros')))
        else:
            model.add(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                           kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros'))

        if is_attention:
            model.add(AttentionWithContext())
            model.add(Addition())

    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_normal(seed=777),
                    bias_initializer='zeros'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    model.summary()

    return model
