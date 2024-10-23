import unicodedata
import re


def preprocess_ja(text):
    text = text.replace("……", "…")
    text = text.replace("─", "ー")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\d[\d,\.]*)", r"#\1#", text)
    text = re.sub(
        r"([０-９！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～、。.！？「」『』（）【】〔〕〈〉《》〘〙〚〛〝〟・゠＝…ー〜ー々〆〇〻])",
        r" \1 ",
        text,
    ).strip()
    text = re.sub(
        r"([0-9!\"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~…–—“”‘’«»])", r" \1 ", text
    ).strip()
    text = re.sub(r"\s+", " ", text)

    return text.lower()


from transformers import AutoTokenizer

ja_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

import py_vncorenlp

rdrsegmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"], save_dir="D:/Project/main_py/javi-dict/VnCoreNLP"
)


def preprocess_vi(text):
    text = text.replace("……", "...")
    text = text.replace("…", "...")
    text = text.replace("─", "-")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("``", '"')
    text = text.replace("''", '"')
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\d[\d,\.]*)", r"#\1#", text)
    text = re.sub(
        r"([０-９！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～、。.！？「」『』（）【】〔〕〈〉《》〘〙〚〛〝〟・゠＝…ー〜ー々〆〇〻])",
        r" \1 ",
        text,
    ).strip()
    text = re.sub(
        r"([0-9!\"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~…–—“”‘’«»])", r" \1 ", text
    ).strip()
    text = re.sub(r"\s+", " ", text)

    return " ".join(rdrsegmenter.word_segment(text.lower()))


vi_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

vi_tokenizer.bos_token_id = 2
vi_tokenizer.cls_token_id = 2
vi_tokenizer.eos_token_id = 3
vi_tokenizer.sep_token_id = 3
vi_tokenizer.unk_token_id = 1
vi_tokenizer.pad_token_id = 0

import tensorflow as tf
import numpy as np


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


class Transformer(tf.keras.Model):
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        dropout_rate=0.1
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1

source_data = np.random.randint(1, 30000, size=(64, 32))
source_data[:, 0] = 2  # Setting the start token at the beginning
source_tensor = tf.constant(source_data, dtype=tf.int64)

target_data = np.random.randint(1, 30000, size=(64, 31))
target_data[:, 0] = 2  # Setting the start token at the beginning
target_tensor = tf.constant(target_data, dtype=tf.int64)

transformer_javi = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=ja_tokenizer.vocab_size,
    target_vocab_size=vi_tokenizer.vocab_size,
    dropout_rate=dropout_rate,
)

output_javi = transformer_javi((source_tensor, target_tensor))

transformer_javi.summary()

transformer_javi.load_weights(
    "D:/Project/main_py/javi-dict/modelWeights/my_model_v3_23_weights.h5"
)

transformer_vija = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=vi_tokenizer.vocab_size,
    target_vocab_size=ja_tokenizer.vocab_size,
    dropout_rate=dropout_rate,
)

output_vija = transformer_vija((source_tensor, target_tensor))

transformer_vija.summary()

transformer_vija.load_weights(
    "D:/Project/main_py/javi-dict/modelWeights/my_model_v1_12_weights.h5"
)
