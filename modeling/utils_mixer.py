from abc import ABCMeta

###
### Mixer
# mixin handlers
class AutoMixinMeta(ABCMeta):
    """
        Helps us conditionally include Mixins, which is useful if we want to switch between different
        combinations of models (ex. SBERT with Doc Embedding, RoBERTa with positional embeddings).

        class Sub(metaclass = AutoMixinMeta):
            def __init__(self, name):
            self.name = name
    """

    def __call__(cls, *args, **kwargs):
        try:
            mixin = kwargs.pop('mixin')
            if isinstance(mixin, list):
                mixin_names = list(map(lambda x: x.__name__, mixin))
                mixin_name = '.'.join(mixin_names)
                cls_list = tuple(mixin + [cls])
            else:
                mixin_name = mixin.__name__
                cls_list = tuple([mixin, cls])

            name = "{}With{}".format(cls.__name__, mixin_name)
            cls = type(name, cls_list, dict(cls.__dict__))
        except KeyError:
            pass
        return type.__call__(cls, *args, **kwargs)


class Mixer(metaclass = AutoMixinMeta):
    """ Class to mix different elements in.

            model = Mixer(config=config, mixin=[SBERTMixin, BiLSTMMixin, TransformerBase])
    """
    pass

