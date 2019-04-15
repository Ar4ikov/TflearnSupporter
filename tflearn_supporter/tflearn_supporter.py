# | Created by Ar4ikov
# | Время: 13.04.2019 - 22:54

from __future__ import absolute_import, print_function, division

from pprint import pprint as print
from importlib import import_module
from json import dumps, loads
from time import time
from uuid import uuid4 as uuid

import tensorflow as tf
import tflearn

import numpy as np
from os import path, listdir, mkdir
from PIL import Image


class TFLS_Error(Exception): ...


class TfLearnSupporter:

    def __init__(self, model_name):
        """
        Что представляет из себя данный класс?
        - Более точное понимание того, к какой архитектуре принадлежит текущие данные уже обученной нейросети
        - Создание своей архитектуры нейросети без написания Python-кода
        - Присвоение данных с уже имеющимся архитектурам
        - Хранение архитектур сети и их дальнейшая подгрузка или выгрузка в JSON-формате
        """
        self.vocab = {}
        self.model_name = model_name

        # Немного о наименовании модели
        self.ext = ".model"
        self.config_file_name = "settings"

        self.vocab.update(model_name=self.model_name)
        self.vocab.update(path_to_model="{}{}".format(
            self.model_name + "/",
            self.model_name + self.ext
        ))

        # Рантайм запуски
        self.run_id = 0
        self.run_uuid = None
        self.last_train_date = None

        self.vocab.update(
            run_id=self.run_id,
            run_uuid=self.run_uuid,
            last_train_date=self.last_train_date
        )

        self.dir_name = self.model_name + "/"

        # Парочка основных переменных
        self.net, self.model = None, None

    @property
    def core(self):
        return tflearn

    @property
    def models(self):
        return tflearn.models

    @property
    def layers(self):
        return tflearn.layers

    @property
    def utils(self):
        return tflearn.data_utils

    @staticmethod
    def load_dataset(name):
        return import_module("tflearn.datasets.{}".format(name))

    @property
    def config(self):
        return tflearn.config

    @staticmethod
    def load_conv_2d_dataset(path_to_dataset, size, channel="RGB"):
        """
        Подгрузка дата-сета для Convolution2D-входа (4D-Tensor)
        :param path_to_dataset: Путь до дата-сета
        :param size: Размер изображения (width x height)
        :param channel: Цветовой канал (RGB, L, 1 & etc.)
        :return: x, y, vocab_
        """
        x, y = [], []

        if channel == "RGB":
            _3rd = 3
        else:
            _3rd = 1

        path_to_dataset = path_to_dataset + "/" if not path_to_dataset.endswith("/") else path_to_dataset

        vocab_ = [x for x in listdir(path_to_dataset)]

        for dir_ in listdir(path_to_dataset):
            for file in listdir(path_to_dataset + dir_):
                img = Image.open(path_to_dataset + dir_ + "/" + file)
                img = img.resize(size)
                img = img.convert(channel)

                img = np.asarray(img, dtype="float")

                y_ = np.zeros(shape=[len(vocab_)])
                x.append(img)
                y_[vocab_.index(dir_)] = 1.

                y.append(y_)

        x = np.asarray(x, dtype="float")
        x = x.reshape([-1, size[1], size[0], _3rd])

        return x, y, vocab_

    @staticmethod
    def shuffle(arrs):
        return tflearn.data_utils.shuffle(arrs)

    def __getattr__(self, item):
        """
        Що цэ делает? Вовзаращает мета-класс, который является нашим "обёрнутым" слоем.
        :param item: - название слоя (аттрибут, который мы хотели вызвать из корневого класса
        :return: метакласс TfLayer
        """
        if getattr(tflearn, item, None):
            class TfLayer:
                def __init__(self, item, root_class: TfLearnSupporter):
                    self.item = item
                    self.root = root_class
                    self.args, self.kwargs = None, None

                    self.sort_func = {"layer": 0, "args": 1, "params": 2}

                def __call__(self, *args, **kwargs):
                    """Вызов метода, полученного через getattr"""
                    self.args, self.kwargs = list(args), {k: v for k, v in kwargs.items()}
                    if "incoming" in self.kwargs:
                        self.kwargs.pop("incoming")

                    # TODO: Fix it
                    for arg in self.args:
                        if isinstance(arg, tflearn.variables.ops.Tensor):
                            self.args.remove(arg)
                            kwargs.update(incoming=arg)

                    self.__dict__()

                    return getattr(tflearn, self.item, None)(*self.args, **kwargs)

                def __dict__(self):
                    """
                    Обновление словаря у родительского класса
                    :return:
                    """
                    if "layers" not in self.root.vocab:
                        self.root.vocab.update(layers=[])
                    self.root.vocab.get("layers").append({z[0]: z[1] for z in sorted([[k, v] for k, v in {
                        "layer": self.item,
                        "args": self.args,
                        "params": self.kwargs
                    }.items()], key=lambda x: self.sort_func[x[0]])})

                    return {"args": self.args, "params": self.kwargs}

            # Обновляем, добавляя новый слой
            self.net = TfLayer(item, self)

            return self.net

    def __dict__(self):
        return self.vocab

    def write_out(self, filename=""):
        if not path.isdir(self.model_name):
            mkdir(self.model_name)

        cfg_filename = path.splitext(filename)[0] if len(filename) > 0 else self.config_file_name

        with open(self.dir_name + cfg_filename + ".json", "w") as f:
            f.write(str(dumps(self.__dict__())))

    def build(self, filename=None, model=tflearn.DNN, tensorboard_verbose=0):
        """
        Быстрое создание сети и модели
        :param filename: Имя файла-кофигуратора архитектуры нейросети
        :param model: Класс модели (DNN, SequenceGenerator & etc.)
        :param tensorboard_verbose: Индекс вывода графиков в Tensorboard
        :return: Net and Model
        """

        self.build_net(filename)
        self.build_model(model_cls=model, tensorboard_verbose=tensorboard_verbose)

        return self.net, self.model

    def build_net(self, filename=None):
        """
        Подгрузка нейросети прямо из директории
        :param filename: имя файла с конфигурацией нейросети и её архитектурой
        :return: нейросети (также подгружается в параметры родительского класса)
        """
        filename = self.dir_name + filename if filename else self.dir_name + self.config_file_name + ".json"

        with open(filename, "r") as f:
            scheme = loads(f.read().replace("\n", ""))

        net = getattr(
            self,
            [x for x in scheme.get("layers") if x.get("layer") == "input_data"][0].get("layer"),
            None)(*[x for x in scheme.get("layers") if x.get("layer") == "input_data"][0].get("args"),
                  **[x for x in scheme.get("layers") if x.get("layer") == "input_data"][0].get("params"))

        scheme.get("layers").remove([x for x in scheme.get("layers") if x.get("layer") == "input_data"][0])

        for layer in scheme.get("layers"):
            _kwargs = layer.get("params")
            _kwargs.update(incoming=net)
            net = getattr(self, layer.get("layer"), None)(*layer.get("args"),
                                                          **_kwargs)
        self.net = net

        self.run_id = scheme.get("run_id")
        self.run_uuid = scheme.get("run_uuid")
        self.last_train_date = scheme.get("last_train_date")

        self.vocab.update(run_id=self.run_id)
        self.vocab.update(run_uuid=self.run_uuid)
        self.vocab.update(last_train_date=self.last_train_date)
        self.vocab.update(tensorflow_version=tf.__version__)

        self.model_name = scheme.get("model_name")

        return self.net

    def build_model(self, model_cls=tflearn.DNN, **kwargs):
        if self.net is None:
            raise TFLS_Error(
                "У вас не загружена ни одна архитектура нейронной сети. Создайте ещё или загрузите уже существующую")

        if "tensorboard_verbose" not in kwargs:
            kwargs.update(tensorboard_verbose=0)

        # Создаём или обновляем наш кофигуратор для данной модели
        self.write_out()

        # Создаём нашу модель
        self.model = model_cls(self.net, **kwargs)

        if path.isfile(self.dir_name + self.model_name + self.ext + ".index"):
            self.model.load(self.dir_name + self.model_name + self.ext)

        return self.model

    def predict(self, x, multiple=False):
        if not self.model:
            raise TFLS_Error(
                "Вы не создали модель нейросети. Предсказания на случайных данных? Неплохо."
            )

        if multiple:
            return self.model.predict(x)
        else:
            return self.model.predict([x])

    @staticmethod
    def get_random_train_set(x, y, percent=0.01):
        """
        Получение случайной контрольной выборки из тренировочной выборки
        :param x: Входные тренировочные данные
        :param y: Выходные тренировочные данные
        :param percent: Процент контрольных данных от тренировочных
        :return: Вход. данные и Выход. данные
        """
        x_test, y_test = [], []

        for i in range(int(len(x) * percent)):
            rand_ = np.random.randint(0, len(x) - 1)

            x_test.append(x[rand_])
            y_test.append(y[rand_])

            x_test.pop(rand_)
            y_test.pop(rand_)

        return x_test, y_test

    def train(self, x, y, validation_set=None, batch_size=64, n_epoch=20, show_metric=True, snapshot_epoch=True,
              snapshot_step=None, shuffle=False, save_every_epoch=True, every_new_run=True):

        """
        Обучение нейронной сети
        :param x: -| Треннировочная выборка на вход
        :param y: -| Треннировочная выборка на выход
        :param validation_set: Контрольная выборка на вход (float or list)
        :param batch_size: Размер выборок за один шаг
        :param n_epoch: Количество эпох обучения (не путать с шагами)
        :param show_metric: Показывать метрические расчёты (точность, например)
        :param snapshot_epoch: Сохранять каждую эпоху для гарантированного правильного сохранения результатов обучения
        :param snapshot_step: Шаг, через который стоит создавать снап
        :param shuffle: Перемешать выборки
        :param save_every_epoch: bool (True or False) - сохранение модели после каждой эпохи или непрерывное обучение
        :param every_new_run: bool (True or False) - запуск каждого этапа обучения, как нового
        :return: Модель tflearn.models
        """

        if not self.model:
            raise TFLS_Error(
                "Вы не создали модель нейросети. Зачем обучать воздух, верно?"
            )

        if every_new_run:
            self.run_id += 1
            self.run_uuid = str(uuid())
            self.vocab["run_id"] += self.run_id
            self.vocab["run_uuid"] = self.run_uuid

        self.last_train_date = int(str(time())[:10])
        self.vocab["last_train_date"] = self.last_train_date

        if save_every_epoch:

            tflearn.is_training(True, session=self.model.session)

            for epoch in range(n_epoch):
                self.model.fit(
                    x, y, validation_set=validation_set, batch_size=batch_size, n_epoch=1, show_metric=show_metric,
                    snapshot_epoch=snapshot_epoch, snapshot_step=snapshot_step, shuffle=shuffle,
                    run_id=self.model_name + "_{}".format(self.run_id)
                )
                self.model.save(self.dir_name + self.model_name + self.ext)
                self.write_out()
            tflearn.is_training(False, session=self.model.session)

        else:

            tflearn.is_training(True, session=self.model.session)

            self.model.fit(
                x, y, validation_set=validation_set, batch_size=batch_size, n_epoch=n_epoch, show_metric=show_metric,
                snapshot_epoch=snapshot_epoch, snapshot_step=snapshot_step, shuffle=shuffle,
                run_id=self.model_name + "_{}".format(self.run_id)
            )
            self.model.save(self.dir_name + self.model_name + self.ext)

            self.write_out()
            tflearn.is_training(False, session=self.model.session)

        return self.model
