from threading import Lock
from flask_socketio import SocketIO
from flask import request
import threading
import time
import tensorflow as tf
from flaskr.utils.model import (
    preprocess_japanese_sentence,
    preprocess_vietnamese_sentece,
    tokenizers,
    split_japanese_sentence,
    split_vietnamese_sentence,
    javi_transformer,
    vija_transformer,
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
                    sentences = split_japanese_sentence(content)

                    for sentence in sentences:
                        sentence = preprocess_japanese_sentence(sentence)
                        sentence = tf.constant(sentence)
                        sentence = tokenizers.ja.tokenize(sentence).to_tensor()
                        encoder_input = sentence

                        start_end = tokenizers.vi.tokenize([""])[0]
                        start = start_end[0][tf.newaxis]
                        end = start_end[1][tf.newaxis]

                        self.__socket.emit(
                            "translate",
                            tokenizers.vi.lookup(tf.reshape(start, (1, 1)))[0][0]
                            .numpy()
                            .decode("utf-8"),
                            to=sid,
                        )

                        output_array = tf.TensorArray(
                            dtype=tf.int64, size=0, dynamic_size=True
                        )
                        output_array = output_array.write(0, start)

                        for i in tf.range(64):
                            output = tf.transpose(output_array.stack())
                            predictions = javi_transformer(
                                [encoder_input, output], training=False
                            )
                            predictions = predictions[:, -1:, :]
                            predicted_id = tf.argmax(predictions, axis=-1)
                            output_array = output_array.write(i + 1, predicted_id[0])

                            self.__socket.emit(
                                "translate",
                                tokenizers.vi.lookup(tf.reshape(predicted_id, (1, 1)))[
                                    0
                                ][0]
                                .numpy()
                                .decode("utf-8"),
                                to=sid,
                            )

                            if predicted_id == end:
                                break

                            if not self.__is_client_connected(sid):
                                break

                        self.__socket.emit(
                            "translate",
                            tokenizers.vi.lookup(tf.reshape(end, (1, 1)))[0][0]
                            .numpy()
                            .decode("utf-8"),
                            to=sid,
                        )

                        output = tf.transpose(output_array.stack())
                        text = tokenizers.vi.detokenize(output)[0]
                case "vija":
                    sentences = split_vietnamese_sentence(content)

                    for sentence in sentences:
                        sentence = preprocess_vietnamese_sentece(sentence)
                        sentence = tf.constant(sentence)
                        sentence = tokenizers.vi.tokenize(sentence).to_tensor()
                        encoder_input = sentence

                        start_end = tokenizers.ja.tokenize([""])[0]
                        start = start_end[0][tf.newaxis]
                        end = start_end[1][tf.newaxis]

                        self.__socket.emit(
                            "translate",
                            tokenizers.ja.lookup(tf.reshape(start, (1, 1)))[0][0]
                            .numpy()
                            .decode("utf-8"),
                            to=sid,
                        )

                        output_array = tf.TensorArray(
                            dtype=tf.int64, size=0, dynamic_size=True
                        )
                        output_array = output_array.write(0, start)

                        for i in tf.range(64):
                            output = tf.transpose(output_array.stack())
                            predictions = vija_transformer(
                                [encoder_input, output], training=False
                            )
                            predictions = predictions[:, -1:, :]
                            predicted_id = tf.argmax(predictions, axis=-1)
                            output_array = output_array.write(i + 1, predicted_id[0])

                            self.__socket.emit(
                                "translate",
                                tokenizers.ja.lookup(tf.reshape(predicted_id, (1, 1)))[
                                    0
                                ][0]
                                .numpy()
                                .decode("utf-8"),
                                to=sid,
                            )

                            if predicted_id == end:
                                break

                            if not self.__is_client_connected(sid):
                                break

                        self.__socket.emit(
                            "translate",
                            tokenizers.ja.lookup(tf.reshape(end, (1, 1)))[0][0]
                            .numpy()
                            .decode("utf-8"),
                            to=sid,
                        )

                        output = tf.transpose(output_array.stack())
                        text = tokenizers.ja.detokenize(output)[0]

    def start(self):
        worker = threading.Thread(target=self.__translate_worker)
        worker.start()
