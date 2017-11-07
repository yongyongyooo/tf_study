import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

tf.app.flags.DEFINE_string("output_graph",
                           "./workspace/graph.pb",
                           "학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string("output_labels",
                           "./workspace/labels.txt",
                           "학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_string("test_set",
                           './test1/',
                           "이미지 추론할 데이터 셋 경로.")
tf.app.flags.DEFINE_boolean("show_image",
                            True,
                            "이미지 추론 후 이미지를 보여줍니다.")

FLAGS = tf.app.flags.FLAGS


def main(_):
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]

    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        logits = sess.graph.get_tensor_by_name('final_result:0')

        for img in os.listdir(FLAGS.test_set):
            image = tf.gfile.FastGFile(FLAGS.test_set + img, 'rb').read()
            prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})

            # print('=== 예측 결과 ===')
            for i in range(len(labels)):
                name = labels[i]
                score = prediction[0][i]
                print('%s (%.2f%%)' % (name, score * 100))

            if FLAGS.show_image:
                img = mpimg.imread(FLAGS.test_set + img)
                plt.imshow(img)
                plt.show()


if __name__ == "__main__":
    tf.app.run()