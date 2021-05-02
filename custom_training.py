True_w = 3
True_b = 2
NUM_EXAMPLES = 1000

random_xs = tf.random.normal(shape = [NUM_EXAMPLES])

ys = random_xs*True_w + True_b

class Model():
  def __init__(self):
    self.w = tf.Variable(2.0, name = 'kernel')
    self.b = tf.Variable(1.0, name = 'bias')

  def __call__(self, x):
    return self.w*x + self.b

def mean_squared_error(y_true, y_pred):
  return tf.reduce_mean(tf.square(y_pred - y_true))

def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as tape:
    current_loss = mean_squared_error(outputs, model(inputs))

  dw, db = tape.gradient(current_loss, [model.w, model.b])
  model.w.assign_sub(dw*learning_rate)
  model.b.assign_sub(db*learning_rate)

  print(model.w, model.b)

model = Model()
epochs = 20

for epoch in range(epochs):
  train(model, random_xs, ys, 0.1)