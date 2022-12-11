from .resnet import build_resnet


def backbone_factory(name):
    if name == 'resnet50':
        return build_resnet(name='resnet50', pretrained=True)
    else:
        raise NotImplementedError(f"Unknown backbone {name}")
