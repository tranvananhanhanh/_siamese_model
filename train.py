from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import os
from siamese import siamese_model
from predata import train_data
from siamese import siamese_model
from tensorflow.keras.metrics import Precision, Recall


#set up loss and optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt= tf.keras.optimizers.Adam(1e-4)

#establish checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt')
checkpoint= tf.train.Checkpoint(opt=opt,siamese_model=siamese_model)

#train function
@tf.function
def train_step(batch):
    #record  all of operations
    with tf.GradientTape() as tape:
        #get anchor  and posi
        # 

        X= batch[:2]
        #get label
        y=batch[2]
       
        #forward pass
        yhat =siamese_model(X, training=True)
        #caculate loss
        loss=binary_cross_loss(y,yhat)
    print(loss)

    #caculate  gradients
    grad= tape.gradient(loss,siamese_model.trainable_variables)

    #caculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad,siamese_model.trainable_variables))
    return loss

#build training loop
def train(data,EPOCHS):
    #loop thrrough epochs
    for epoch in range(1,EPOCHS+1):
        print('\n Epoch{}/{}'.format(epoch,EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        r=Recall()
        p=Precision()
        #loop through each batch
        for idx, batch in enumerate(data):
            #run train step here
            loss=train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 

            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

        #save checkpoint
        if epoch%10==0:
            checkpoint.save(file_prefix=checkpoint_prefix)

#train the model
EPOCHS = 10
train(train_data,EPOCHS) 
#save model vào tập h5
siamese_model.save('siamesemodel.h5')