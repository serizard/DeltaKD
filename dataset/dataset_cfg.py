class CIFAR10Config:
    NAME = 'cifar-10'
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
    INPUT_SIZE = 224
    NUM_CLASSES = 10
    CROP_PCT = 0.875
    
    # Augmentation config
    RAND_AUGMENT = dict(
        num_ops=2,
        magnitude=0.3
    )
    MIXUP = dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1
    )
    RANDOM_ERASE = dict(
        probability=0.25,
        mode='pixel',
        count=1
    )

class CIFAR100Config:
    NAME = 'cifar-100'
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    INPUT_SIZE = 224  
    NUM_CLASSES = 100
    CROP_PCT = 0.875
    
    # Augmentation config
    RAND_AUGMENT = dict(
        num_ops=2,
        magnitude=0.3
    )
    MIXUP = dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1
    )
    RANDOM_ERASE = dict(
        probability=0.25,
        mode='pixel',
        count=1
    )


class ImageNet1KConfig:
    NAME = 'imagenet-1k'
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    INPUT_SIZE = 224
    NUM_CLASSES = 1000
    CROP_PCT = 0.875
    
    # Augmentation config
    RAND_AUGMENT = dict(
        num_ops=9,
        magnitude=0.5
    )
    MIXUP = dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1
    )
    RANDOM_ERASE = dict(
        probability=0.25,
        mode='pixel',
        count=1
    )


class ImageNet21KConfig:
    NAME = 'imagenet-21k'
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    INPUT_SIZE = 224
    NUM_CLASSES = 21843
    CROP_PCT = 0.875
    
    # Augmentation config
    RAND_AUGMENT = dict(
        num_ops=9,
        magnitude=0.5
    )
    MIXUP = dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1
    )
    RANDOM_ERASE = dict(
        probability=0.25,
        mode='pixel',
        count=1
    )