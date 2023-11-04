import tensorflow as tf
import numpy as np

# Assuming `args` is an object with the necessary attributes
class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        # choose different rnn cell 
        if args.model == 'rnn':
            cell_fn = tf.keras.layers.SimpleRNNCell
        elif args.model == 'gru':
            cell_fn = tf.keras.layers.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.keras.layers.LSTMCell
        # NASCell is not available in tf.keras, you would need to find an alternative or implement it.
        # elif args.model == 'nas':
        #     cell_fn = tf.keras.experimental.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # warp multi layered rnn cell into one cell with dropout
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = tf.keras.layers.Dropout(rate=1 - args.output_keep_prob)(cell)
            cells.append(cell)
        if len(cells) > 1:
            self.cell = cell = tf.keras.layers.StackedRNNCells(cells)
        else:
            self.cell = cell = cells[0]

        self.inputs = tf.keras.Input(shape=(args.seq_length,), batch_size=args.batch_size, dtype=tf.int32)
        
        # embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=args.vocab_size, output_dim=args.rnn_size)
        embedded_inputs = self.embedding_layer(self.inputs)
        
        # build RNN
        if args.model == 'rnn' or args.model == 'gru' or args.model == 'lstm':
            rnn_layer = getattr(tf.keras.layers, args.model.upper())(args.rnn_size, return_sequences=True, stateful=True)
            outputs = rnn_layer(embedded_inputs)

        # you could potentially use `return_state=True` and handle the states manually if you need statefulness

        # softmax output layer
        self.output_layer = tf.keras.layers.Dense(args.vocab_size, activation='softmax')
        logits = self.output_layer(outputs)

        # loss, optimizer and training operation
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

        # compile model
        self.model = tf.keras.Model(inputs=self.inputs, outputs=logits)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        # Summary and checkpoints callbacks
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            tf.keras.callbacks.ModelCheckpoint(filepath='./save/model-{epoch:02d}-{loss:.2f}.hdf5'),
        ]

    # other functions such as sample, etc.
    # ...

# # Model usage example
# args = ... # set your arguments here
# model = Model(args)
# model.model.fit(dataset, epochs=10, callbacks=model.callbacks)
