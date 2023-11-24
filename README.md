수정한 부분
------------mrcnn/model.py 부분
-38~48번 라인
class AnchorsLayer(KL.Layer): #수정 추가
    def __init__(self, anchors, name="anchors", **kwargs):
        super(AnchorsLayer, self).__init__(name=name, **kwargs)
        self.anchors = tf.Variable(anchors)

    def call(self, dummy):
        return self.anchors

    def get_config(self):
        config = super(AnchorsLayer, self).get_config()
        return config

-353번 라인
return tf.math.log(x) / tf.math.log(2.0) #수정

-559번 라인
negative_indices = tf.where(tf.math.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0] #수정

-565번 라인
positive_indices = tf.random.shuffle(positive_indices)[:positive_count] #수정

-570번 라인
negative_indices = tf.random.shuffle(negative_indices)[:negative_count] #수정

-732~734번 라인
keep = tf.sets.intersection(tf.expand_dims(keep, 0), #수정
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0] #수정

-770~772번 라인
keep = tf.sets.intersection(tf.expand_dims(keep, 0), #수정
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0] #수정

-784번 라인
detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.cast(tf.gather(class_ids, keep), dtype=tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)  # 수정

-963~966번 라인
 if s[1] is None: #수정
        mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    else:
        mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

-1949번 라인
anchors = AnchorsLayer(anchors, name="anchors")(input_image) #수정

-2145번 라인
hdf5_format.load_weights_from_hdf5_group_by_name(f, layers) #수정

-2147번 라인
hdf5_format.load_weights_from_hdf5_group(f, layers) #수정

--------------mrcnn/utils.py부분
-202~203번 라인
    dh = tf.math.log(gt_height / height) #수정
    dw = tf.math.log(gt_width / width) #수정

--------------samples/balloon/balloon.py부분
-94번 라인
dataset_dir = "C:/Users/leean/PycharmProjects/opensw_maskrcnn/Mask_RCNN_3/samples/balloon/dataset/train" #수정