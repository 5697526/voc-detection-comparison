_base_ = [
    './_base_/models/mask-rcnn_r50_fpn.py',
    './_base_/datasets/voc0712.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

# 模型配置 - 完全移除mask分支
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),  # VOC有20个类别
        mask_head=None,  # 完全禁用mask分支
        mask_roi_extractor=None  # 同时禁用mask的ROI提取器
    ),
    # 测试配置也移除mask相关设置
    test_cfg=dict(
        rcnn=dict(
            with_mask=False  # 确保测试时不生成mask预测
        )
    )
)

# 数据集设置（确保不加载mask标注）
data = dict(
    train=dict(
        dataset=dict(
            datasets=[
                dict(
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations',
                             with_bbox=True, with_mask=False),
                        dict(type='Resize', scale=(1000, 600), keep_ratio=True),
                        dict(type='RandomFlip', prob=0.5),
                        dict(type='PackDetInputs')
                    ]
                ),
                dict(
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations',
                             with_bbox=True, with_mask=False),
                        dict(type='Resize', scale=(1000, 600), keep_ratio=True),
                        dict(type='RandomFlip', prob=0.5),
                        dict(type='PackDetInputs')
                    ]
                )
            ]
        )
    ),
    val=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]
    ),
    test=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]
    )
)

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# 评估器设置（标准VOC评估，不涉及mask）
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator


# 训练设置
train_cfg = dict(max_epochs=6, val_interval=1)
