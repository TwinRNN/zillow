import tensorflow as tf
import numpy as np
import dataset_zillow
import time


def build_model(pkeep, house_ph, logerror):
    '''
    pkeep = tf.placeholder(
                tf.float32, name='pkeep')  # dropout parameter

    with tf.name_scope('inputs'):
        house_ph = tf.placeholder(
            tf.float32, [None, None, event_feature_size], name='house_ph')
        logerror = tf.placeholder(
            tf.float32, [None, None, 1], name='logerror')
        time_series_days = tf.placeholder(
            tf.float32, [None, None, None, timeseries_feature_size], name='time_series')
        time_series_weeks = tf.placeholder(
            tf.float32, [None, None, None, timeseries_feature_size], name='time_series')
        time_series_months = tf.placeholder(
            tf.float32, [None, None, None, timeseries_feature_size], name='time_series')
    '''

    batch_size = tf.shape(house_ph)[0]
    seq_len = tf.shape(house_ph)[1]

    with tf.variable_scope("house"):
        cells = [tf.contrib.rnn.GRUCell(
            128, bias_initializer=tf.zeros_initializer()) for _ in range(2)]
        cells = [tf.contrib.rnn.DropoutWrapper(
            cell, input_keep_prob=pkeep) for cell in cells]
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)
        # cell = tf.contrib.rnn.AttentionCellWrapper(cell, state_size)
        # init_state = tf.random_normal([batch_size, state_size])
        # init_state = cell.zero_state(batch_size, tf.float32)
        rnn_house_outputs, H_house = tf.nn.dynamic_rnn(
                cell, house_ph, dtype=tf.float32)

    with tf.name_scope("concat"):
        concat_outputs = rnn_house_outputs

    with tf.name_scope("prediction"):
        prediction = tf.contrib.layers.linear(
            concat_outputs, 1)
        prediction = tf.identity(prediction, name='prediction')

    with tf.name_scope("error"):
        price_error = tf.abs(1 - tf.exp(logerror-prediction))
        price_error = tf.reduce_mean(price_error, name='error')
        tf.summary.scalar('price error', price_error)

    # MSE loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(logerror - prediction))
        tf.summary.scalar('loss', loss)
    return prediction, loss


def average_losses(loss):
    tf.add_to_collection('losses', loss)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses')

    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def feed_all_gpu(inp_dict, models, payload_per_gpu, pkeep, house_ph, logerror):
    for i in range(len(models)):
        p,x, y, _, _, _ = models[i]
        start_pos = int(i * payload_per_gpu)
        stop_pos = int((i + 1) * payload_per_gpu)
        inp_dict[p] = pkeep
        inp_dict[x] = house_ph[start_pos:stop_pos]

        inp_dict[y] = logerror[start_pos:stop_pos]
    return inp_dict


def multi_gpu(num_gpu):
    batch_size = 128 * num_gpu

    dataset = dataset_zillow.Dataset(batch_size, 128)
    dataset.split(3600,117600,3600,3600,0)
    event_feature_size = dataset.feature_size()['event']
    #timeseries_feature_size = dataset.feature_size()['time']
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        with tf.device('/cpu:0'):
            learning_rate = tf.placeholder(tf.float32, shape=[])
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

            print('build model...')
            print('build model on gpu tower...')
            models = []
            for gpu_id in range(num_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    print('tower:%d...' % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('cpu_variables', reuse=gpu_id > 0):
                            pkeep = tf.placeholder(
                                tf.float32, name='pkeep')  # dropout parameter
                            house_ph = tf.placeholder(
                                tf.float32, [None, None, event_feature_size], name='house_ph')
                            logerror = tf.placeholder(
                                tf.float32, [None, None, 1], name='logerror')

                            pred, loss = build_model(pkeep, house_ph, logerror)
                            #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
                            grads = opt.compute_gradients(loss)
                            models.append(
                                (pkeep, house_ph, logerror, pred, loss, grads))
            print('build model on gpu tower done.')

            print('reduce model on cpu...')
            _, tower_x,tower_y, tower_preds, tower_losses, tower_grads = zip(*models)
            aver_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = opt.apply_gradients(
                average_gradients(tower_grads))

            all_y = tower_y
            all_pred = tower_preds
            #correct_pred = all_y-all_pred

            #accuracy = tf.reduce_mean(tf.square(correct_pred, name="acc"))
            print('reduce model on cpu done.')

            print('run train op...')
            if 'session' in locals() and sess is not None:
                print('Close interactive session')
            sess.close()
            sess.run(tf.global_variables_initializer())

            lr = 0.01
            for epoch in range(10):
                start_time = time.time()
                payload_per_gpu = batch_size/num_gpu
                total_batch = int(120000/batch_size)
                avg_loss = 0.0
                print('\n---------------------')
                print('Epoch:%d, lr:%.4f' % (epoch, lr))
                for batch_idx in range(total_batch):
                    house_ph, logerror = dataset.next_batch()
                    inp_dict = {}
                    inp_dict[learning_rate] = lr
                    inp_dict = feed_all_gpu(
                        inp_dict, models, payload_per_gpu, 0.8, house_ph, logerror)
                    _, _loss = sess.run(
                        [apply_gradient_op, aver_loss_op], inp_dict)
                    avg_loss += _loss
                avg_loss /= total_batch
                print('Train loss:%.4f' % (avg_loss))

                lr = max(lr * 0.7,0.00001)

                val_payload_per_gpu = 1800
                total_batch = 1
                preds = None
                ys = None
                for batch_idx in range(total_batch):
                    batch_x,batch_y = dataset.valid_data()
                    inp_dict = feed_all_gpu({}, models, val_payload_per_gpu,1.0, batch_x, batch_y)
                    batch_pred,batch_y = sess.run([all_pred,all_y], inp_dict)
                    if preds is None:
                        preds = batch_pred
                    else:
                        preds = np.concatenate((preds, batch_pred), 0)
                    if ys is None:
                        ys = batch_y
                    else:
                        ys = np.concatenate((ys,batch_y),0)
                preds = np.array(preds)
                ys = np.array(ys)
                print(preds,ys)
                val_accuracy = np.mean(np.square(ys-preds))
                print('Val Accuracy: %0.4f%%' % (100.0 * val_accuracy))

                stop_time = time.time()
                elapsed_time = stop_time-start_time
                print('Cost time: ' + str(elapsed_time) + ' sec.')
            print('training done.')

            test_payload_per_gpu = 1800
            total_batch = 1
            preds = None
            ys = None
            for batch_idx in range(total_batch):
                batch_x, batch_y = dataset.test_data()
                inp_dict = feed_all_gpu({}, models, test_payload_per_gpu, 1.0,batch_x, batch_y)
                batch_pred, batch_y = sess.run([all_pred, all_y], inp_dict)
                if preds is None:
                    preds = batch_pred
                else:
                    preds = np.concatenate((preds, batch_pred), 0)
                if ys is None:
                    ys = batch_y
                else:
                    ys = np.concatenate((ys, batch_y), 0)
            preds = np.array(preds)
            ys = np.array(ys)
            print(preds,ys)
            test_accuracy = np.mean(np.square(ys-preds))
            print('Test Accuracy: %0.4f%%\n\n' % (100.0 * test_accuracy))

if __name__ == "__main__":
    multi_gpu(1)
