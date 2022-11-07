import paddle
import paddle.nn as nn
import paddle.nn.initializer as I

from paddlenlp.embeddings import TokenEmbedding


class BiLSTM(nn.Layer):

    def __init__(self,
                 embed_dim,
                 hidden_size,
                 vocab_size,
                 output_dim,
                 vocab_path,
                 padding_idx=0,
                 num_layers=1,
                 dropout_prob=0.0,
                 init_scale=0.1,
                 embedding_name=None):
        super(BiLSTM, self).__init__()
        if embedding_name is not None:
            self.embedder = TokenEmbedding(embedding_name,
                                           extended_vocab_path=vocab_path,
                                           keep_extended_vocab_only=True)
            embed_dim = self.embedder.embedding_dim
        else:
            self.embedder = nn.Embedding(vocab_size, embed_dim, padding_idx)

        self.lstm = nn.LSTM(embed_dim,
                            hidden_size,
                            num_layers,
                            'bidirectional',
                            dropout=dropout_prob)

        self.fc = nn.Linear(
            hidden_size * 2,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

        self.fc_1 = nn.Linear(
            hidden_size * 8,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

        self.output_layer = nn.Linear(
            hidden_size*2,
            output_dim,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

    def forward(self, x, seq_len):
        x_embed = self.embedder(x)
        lstm_out, (hidden_1, _) = self.lstm(x_embed, sequence_length=seq_len)
        logits = self.output_layer(lstm_out)

        return logits
