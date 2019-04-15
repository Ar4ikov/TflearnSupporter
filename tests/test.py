# | Created by Ar4ikov
# | Время: 15.04.2019 - 23:38

import unittest
from importlib import import_module


class TFLS_Tests(unittest.TestCase):
    """
    Прежде чем начать использовать это, посетите:
    http://tflearn.org
    """

    def test_tf_support(self):
        tf = import_module("tensorflow")

        self.assertIsNotNone(tf.__version__)

    def test_write_out(self):
        tfls = import_module("tflearn_supporter")

        print(tfls.__version__)

        tfls_cls = tfls.TfLearnSupporter(model_name="test")

        # Тестовая нейросеть
        net = tfls_cls.input_data(shape=[None, 1])
        net = tfls_cls.fully_connected(net, 1)

        # Аутпут файла
        tfls_cls.write_out()

        # Подгрузка сети из файла
        tfls_cls_1 = tfls.TfLearnSupporter(model_name="test")
        self.assertIsNotNone(tfls_cls_1.build_net())

        # Аутпут фалйа с другим названием
        tfls_cls_1.write_out(filename="net.json")

        # Подгрузка файла с другим названием
        tfls_cls_2 = tfls.TfLearnSupporter(model_name="test")
        self.assertIsNotNone(tfls_cls_2.build_net(filename="net.json"))

        # Удаление тестовой директории
        from shutil import rmtree

        # rmtree("test")

    def test_kek(self):
        pass


if __name__ == "__main__":
    unittest.main()
