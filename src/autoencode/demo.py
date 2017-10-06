from AdditiveGaussianNoiseAutoencoder import *

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
X_train,X_test=standard_scale(mnist.train.images, mnist.test.images)
n_samples=int(mnist.train.num_examples)
training_epochs=20
batch_size=128
display_step=1

autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=200,
                                             transfer_function=tf.nn.softplus,
                                             optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)
for epoch in range(training_epochs):
    avg_cost=0.
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train,batch_size)
        
        
        cost=autoencoder.partial_fit(batch_xs)
        avg_cost+=cost/n_samples*batch_size
        
    if epoch%display_step==0:
        print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))


print("Total cost:"+str(autoencoder.calc_total_cost(X_test)))  