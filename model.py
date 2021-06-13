from tensorflow.keras import layers, Model

# Builda o modelo
def build_model (state_shape, actions_n):
  inputs = layers.Input(state_shape)

  layer1 = layers.Conv2D(32, 4, strides=3, activation="relu")(inputs)
  layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
  layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

  layer4 = layers.Flatten()(layer3)

  layer5 = layers.Dense(state_shape[0] * state_shape[1], activation="relu")(layer4)
  action = layers.Dense(actions_n, activation="linear")(layer5)

  model = Model(inputs=inputs, outputs=action)

  return model