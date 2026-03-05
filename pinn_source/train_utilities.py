import tensorflow as tf
import nisaba as ns

def lagrange_callback(pb, itr, itr_round, gamma, lambda_0, eta, hloss_names = [], frequency = 200):
    if itr % frequency == 0 and itr > 0:
        # update HLosses
        for loss in pb.losses:
            if loss.name in hloss_names:
                # update Lagrange multiplier
                lambda_old = loss.lagrange_mul # tf.gather(loss.lagrange_mul, indices)
                lambda_new = gamma * lambda_old + eta * loss.normalized_values(pb.data.current_batch)

                delta_lambda = (lambda_new - lambda_old) / (lambda_old + 1e-9)
                delta_lambda_avg = tf.reduce_mean(tf.abs(delta_lambda))
                pb.history['losses'][loss.name]['log_h'].append(delta_lambda_avg.numpy())

                loss.lagrange_mul.assign(lambda_new + lambda_0)

def curriculum_loss_callback(pb, itr, itr_round, pde_weight, epsilon, frequency = 10):
    if itr % frequency == 0 and itr > 0:
        loss_values = [0. for _ in range(10)]
        for loss in pb.losses:
            name = loss.name
            if len(name) > 3 and name[:3] == 'PDE':
                number = int(name[4])
                loss_values[number] = loss.loss_base_call([])

        weights = [tf.constant(1., dtype=ns.config.get_dtype())]
        for i in range(1, 10):
            weights.append(tf.math.exp(-epsilon * tf.reduce_mean(loss_values[0:i])))

        for loss in pb.losses:
            name = loss.name
            if len(name) > 3 and name[:3] == 'PDE':
                number = int(name[4])
                loss.weight = tf.expand_dims(pde_weight * weights[number], -1)

                pb.history['weights'][loss.name]['log'].append(weights[number].numpy().item())
                print(loss.weight, loss_values[number])

def weight_adjustment_callback(pb, itr, itr_round, model, alpha = 0.5, frequency = 10):
    if itr % frequency == 0 and itr > 0:
        norms = []
        norm_sum = 0.
        for loss in pb.losses:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.trainable_variables)
                loss_value = loss.loss_base_call([])
                #print(loss_value)
                grads = tape.gradient(loss_value, model.trainable_variables)
                gradient_norm = tf.constant(0, dtype=ns.config.get_dtype())
    
                for gradient in grads:
                    if gradient is not None:
                        gradient_norm += tf.reduce_sum(tf.square(gradient))
                gradient_norm = tf.sqrt(gradient_norm)
                norms += [(gradient_norm, loss)]
                norm_sum += gradient_norm
        
        for (gradient_norm, loss) in norms:
            if gradient_norm > 0.:
                loss.weight = alpha * norm_sum / gradient_norm + (1-alpha) * loss.weight
                pb.history['weights'][loss.name]['log'].append(loss.weight.numpy().item())
            #print(norm_sum, gradient_norm, loss.weight)

def relobralo_callback(pb, itr, itr_round, variables, alpha, T, bernoulli_vec):
    m = len(pb.losses)
    if m > 1 and itr > 3:
        rho = bernoulli_vec[itr]
        loss_comp_last = [pb.history['losses'][loss.name]['log'][-1]/(T*pb.history['losses'][loss.name]['log'][-2]) for loss in pb.losses]
        weights = (1-alpha) * m * tf.nn.softmax(loss_comp_last)
        if rho == 1:
            weights += alpha * pb.history['losses'][loss.name]['log'][-2]
        else:
            loss_comp_initial = [pb.history['losses'][loss.name]['log'][-1]/(T*pb.history['losses'][loss.name]['log'][0]) for loss in pb.losses]
            weights += alpha * tf.nn.softmax(loss_comp_initial)
        for i, loss in enumerate(pb.losses):
            loss.weight = weights[i]
            pb.history['weights'][loss.name]['log'].append(loss.weight.numpy().item())

def save_models(models, filenames):
    
    for model, file in zip(models, filenames):
        model.save(file)

def setup_callbacks(params, filename, filename_history, model, np_random_generator, frequency = 10, weight_PDE = 1.):
    pltcb = ns.utils.HistoryPlotCallback(gui = False, filename = filename, filename_history = filename_history, frequency = frequency)
    callbacks_orig = [pltcb]
    callbacks = [pltcb]
    if params['adapt']:
        weightcb = lambda pb, itr, itr_round: weight_adjustment_callback(pb, itr, itr_round, model)
        callbacks += [weightcb]
    elif params['relobralo']:
        n = max([params['adam1'], params['adam2']])
        bernoulli_vec = np_random_generator.binomial(1, params['relobralo_params']['p_back'], n)
        weightcb = lambda pb, itr, itr_round: relobralo_callback(pb, itr, itr_round, model.trainable_variables, params['relobralo_params']['alpha'], params['relobralo_params']['T'], bernoulli_vec)
        callbacks += [weightcb]
    
    if 'RBA' in params and params['RBA']:
        rbacb = lambda pb, itr, itr_round: lagrange_callback(pb, itr, itr_round, params['RBA_params']['gamma'], params['RBA_params']['lambda0'], params['RBA_params']['eta'], params['RBA_params']['losses'], frequency = params['RBA_params']['update_frequency'])
        callbacks += [rbacb]
        
    if 'eps' in params and params['eps'] > 0.:
        curriculum_cb = lambda pb, itr, itr_round: curriculum_loss_callback(pb, itr, itr_round, weight_PDE, params['eps'])
        callbacks += [curriculum_cb]

    return callbacks, callbacks_orig, pltcb

def get_param_function(key, **kwargs):
    if key == 'exp':
        param_func = lambda x: tf.math.exp(x)
        inv_param_func = lambda x: tf.math.log(x)
    elif key == 'square':
        param_func = lambda x: tf.math.pow(x, 2)
        inv_param_func = lambda x: tf.math.sqrt(x)
    elif key == 'square2':
        param_func = lambda x: 0.5*tf.math.pow(x, 2)
        inv_param_func = lambda x: tf.math.sqrt(2*x)
    elif key == 'abs':
        param_func = lambda x: tf.math.abs(x, 2)
        inv_param_func = lambda x: tf.math.abs(x)
    elif key == 'sigmoid':
        min_val = kwargs['min'] if 'min' in kwargs else 0
        max_val = kwargs['max'] if 'max' in kwargs else 1
        min_val = tf.expand_dims(tf.constant(min_val, dtype=ns.config.get_dtype()),-1)
        max_val = tf.expand_dims(tf.constant(max_val, dtype=ns.config.get_dtype()),-1)

        logit = lambda x: tf.math.log(tf.math.divide(x, 1-x))
        
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 1.0
        
        param_func = lambda x: tf.math.add(tf.math.multiply(max_val - min_val, tf.math.sigmoid(a*x)), min_val)
        inv_param_func = lambda x: logit(tf.math.divide(x-min_val, max_val-min_val))/a
    else:
        param_func = lambda x: x
        inv_param_func = lambda x: x

    return param_func, inv_param_func
