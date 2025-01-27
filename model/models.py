from timm.models import (deit_base_patch16_224, deit_small_patch16_224, deit_tiny_patch16_224,
                         swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224)


def get_model(model_name, num_classes=1000, pretrained=True):
    if model_name == 'deit_base':
        return deit_base_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'deit_small':
        return deit_small_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'deit_tiny':
        return deit_tiny_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'swin_tiny':
        return swin_tiny_patch4_window7_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'swin_small':
        return swin_small_patch4_window7_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'swin_base':
        return swin_base_patch4_window7_224(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not found")
