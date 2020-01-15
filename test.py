import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_list("a", None, "a list")

FLAGS = flags.FLAGS

def main(args):
    print(FLAGS.a)

if __name__ == "__main__":
    print(tf.__version__)
    tf.app.run()
