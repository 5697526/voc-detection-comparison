# custom_dataset.py
_base_ = './voc0712.py'

# 修改数据集路径
data_root = 'data/'
data = dict(
    test=dict(
        data_root=data_root,
        ann_file=None,  # 无标注文件
        img_prefix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                scale=(1333, 800),
                keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=False),  # 无标注，不加载标注
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]
    )
)
