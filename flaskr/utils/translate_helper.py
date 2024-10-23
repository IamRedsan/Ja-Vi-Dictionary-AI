from threading import Lock
from flask_socketio import SocketIO
from flask import request
import threading
import time
import tensorflow as tf
from flaskr.utils.model import (
    preprocess_vi,
    preprocess_ja,
    transformer_javi,
    transformer_vija,
    ja_tokenizer,
    vi_tokenizer,
)


class TranslateManager:

    __instance = None

    @staticmethod
    def get_instance():
        if TranslateManager.__instance is None:
            TranslateManager()
        return TranslateManager.__instance

    def __init__(self):
        if TranslateManager.__instance is not None:
            raise Exception("This class is a singleton!")

        self.__socket = SocketIO()

        def handle_translate_request(data):
            if (
                data is None
                or "content" not in data
                or not isinstance(data["content"], str)
                or len(data["content"].strip()) == 0
                or "model" not in data
                or data["model"] not in ["javi", "vija"]
            ):
                return

            self.__add_translate_queue(request.sid, data["model"], data["content"])

        self.__socket.on_event("translate-request", handle_translate_request)

        def handle_connect():
            self.__add_client(request.sid)

        self.__socket.on_event("connect", handle_connect)

        def handle_disconnect():
            self.__remove_client(request.sid)

        self.__socket.on_event("disconnect", handle_disconnect)

        self.__client_sids = {}
        self.__client_sids_lock = Lock()

        self.__translate_queue = []
        self.__translate_queue_lock = Lock()

        TranslateManager.__instance = self

    def init_app(self, app):
        self.__socket.init_app(app)

    def __add_client(self, sid):
        with self.__client_sids_lock:
            self.__client_sids[sid] = True

    def __remove_client(self, sid):
        with self.__client_sids_lock:
            del self.__client_sids[sid]

    def __is_client_connected(self, sid):
        with self.__client_sids_lock:
            return self.__client_sids[sid]

    def __add_translate_queue(self, sid, model, content):
        with self.__translate_queue_lock:
            self.__translate_queue.append(
                {"sid": sid, "model": model, "content": content}
            )

    def __pop_translate_queue(self):
        with self.__translate_queue_lock:
            if len(self.__translate_queue) == 0:
                return None
            return self.__translate_queue.pop(0)

    def __translate_worker(self):
        while True:
            time.sleep(1)

            job = self.__pop_translate_queue()
            if job is None:
                continue

            sid = job["sid"]
            model = job["model"]
            content = job["content"]

            match model:
                case "javi":
                    encoder_input = tf.expand_dims(
                        tf.cast(
                            ja_tokenizer(preprocess_ja(content))["input_ids"], tf.int64
                        ),
                        axis=0,
                    )
                    start_end = vi_tokenizer("")["input_ids"]
                    start = tf.expand_dims(tf.cast(start_end[0], tf.int64), axis=0)
                    end = tf.expand_dims(tf.cast(start_end[1], tf.int64), axis=0)
                    output_array = tf.TensorArray(
                        dtype=tf.int64, size=0, dynamic_size=True
                    )
                    output_array = output_array.write(0, start)

                    self.__socket.emit(
                        "translate",
                        vi_tokenizer.decode(start.numpy()),
                        to=sid,
                    )

                    for i in range(128):

                        if not self.__is_client_connected(sid):
                            break

                        output = tf.transpose(output_array.stack())
                        predictions = transformer_javi(
                            [encoder_input, output], training=False
                        )
                        predictions = predictions[:, -1:, :]
                        predicted_id = tf.argmax(predictions, axis=-1)
                        output_array = output_array.write(i + 1, predicted_id[0])

                        self.__socket.emit(
                            "translate",
                            vi_tokenizer.decode(predicted_id[0].numpy()),
                            to=sid,
                        )

                        if predicted_id[0] == end:
                            break
                case "vija":
                    encoder_input = tf.expand_dims(
                        tf.cast(
                            vi_tokenizer(preprocess_vi(content))["input_ids"], tf.int64
                        ),
                        axis=0,
                    )
                    start_end = vi_tokenizer("")["input_ids"]
                    start = tf.expand_dims(tf.cast(start_end[0], tf.int64), axis=0)
                    end = tf.expand_dims(tf.cast(start_end[1], tf.int64), axis=0)
                    output_array = tf.TensorArray(
                        dtype=tf.int64, size=0, dynamic_size=True
                    )
                    output_array = output_array.write(0, start)

                    self.__socket.emit(
                        "translate",
                        ja_tokenizer.decode(start.numpy()),
                        to=sid,
                    )

                    for i in range(128):

                        if not self.__is_client_connected(sid):
                            break

                        output = tf.transpose(output_array.stack())
                        predictions = transformer_vija(
                            [encoder_input, output], training=False
                        )
                        predictions = predictions[:, -1:, :]
                        predicted_id = tf.argmax(predictions, axis=-1)
                        output_array = output_array.write(i + 1, predicted_id[0])

                        self.__socket.emit(
                            "translate",
                            ja_tokenizer.decode(predicted_id[0].numpy()),
                            to=sid,
                        )

                        if predicted_id[0] == end:
                            break

    def start(self):
        worker = threading.Thread(target=self.__translate_worker)
        worker.start()
