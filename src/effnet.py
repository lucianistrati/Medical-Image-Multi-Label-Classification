from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import Adam
import pandas as pd

# define input shape and batch size
input_shape = (224, 224, 3)
batch_size = 256

# paths
train_dir = "data/train_images"
test_dir = "data/val_images" # didn't work with ImageDataGeneartor.flow_from_dataframe
# test_dir = "data/test_images"
csv_dir = "data/train_labels.csv"
label_names_dir = "data/categories.csv"

# read csv data for loading image label information
df = pd.read_csv(csv_dir)
df_labels = pd.read_csv(label_names_dir)

label_names = list(df_labels["Labels"])
x_col = df.columns[0]
y_cols = list(df.columns[1:len(label_names)+1])

# load input images and split into training, test and validation
datagen = ImageDataGenerator(rescale=1./255,validation_split=.25)

train_generator = datagen.flow_from_dataframe(
    df,
    directory=train_dir,
    x_col=x_col,
    y_col=y_cols,
    subset="training",
    target_size=input_shape[0:2],
    color_mode="rgb",
    class_mode="raw", # for multilabel output
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
    validate_filenames=False
)

test_generator = datagen.flow_from_dataframe(
    df,
    directory=train_dir,
    x_col=x_col,
    y_col=y_cols,
    subset="validation",
    target_size=input_shape[0:2],
    color_mode="rgb",
    class_mode="raw",
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
    validate_filenames=False
)

# build model
n_outputs = len(label_names)

model_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
model_base.trainable = False

model = Sequential([
    model_base,
    Dropout(0.25),
    Flatten(),
    Dense(n_outputs, activation="sigmoid")
])

model.summary()

# compile model
opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# define training and validation steps
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = test_generator.samples // test_generator.batch_size

# train model
hist = model.fit(
    train_generator,
    epochs=100, steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=validation_steps).history

print(hist.keys())
