"""Factory module for optimizer factory.

copyright Matti Kaupenjohann, 2022
"""

from torch.optim import Adam, SGD

from boxsup_pytorch.core import object_factory


class OptimizerProvider(object_factory.ObjectFactory):
    def get(self, service_id, **kwargs):
        return self.create(service_id, **kwargs)


class AdamBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, params, lr, **kwargs):
        if not self._instance:
            self._instance = Adam(params, lr, **kwargs)
        return self._instance


class SGDBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, params, lr, **kwargs):
        if not self._instance:
            self._instance = SGD(params, lr, **kwargs)
        return self._instance


provider = OptimizerProvider()
provider.register_builder("ADAM", AdamBuilder)
provider.register_builder("SGD", SGDBuilder)
