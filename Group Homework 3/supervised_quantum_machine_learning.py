import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
import tensorflow as tf
from qmlt.tf.helpers import make_param
from qmlt.tf import CircuitLearner

def circuit(X):
    phi = make_param('phi', constant=2.)

    eng, q = sf.Engine(2)

    with eng:
        Dgate(X[:, 0], 0.) | q[0]
        Dgate(X[:, 1], 0.) | q[1]
        BSgate(phi=phi) | (q[0], q[1])
        BSgate() | (q[0], q[1])

    num_inputs = X.get_shape().as_list()[0]
    state = eng.run('tf', cutoff_dim=10, eval=False, batch_size=num_inputs)

    p0 = state.fock_prob([0, 2])
    p1 = state.fock_prob([2, 0])
    normalisation = p0 + p1 + 1e-10
    circuit_output = p1/normalisation

    return circuit_output

def myloss(circuit_output, targets):
    return tf.losses.mean_squared_error(labels=circuit_output, predictions=targets)

X_train = [[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]]
Y_train = [1, 1, 0, 0]
X_test = [[0.25, 0.5], [0.5, 0.25]]
Y_test = [1, 0]
X_pred = [[0.4, 0.5], [0.5, 0.4]]

def outputs_to_predictions(circuit_output):
    return tf.round(circuit_output)

hyperparams = {'circuit': circuit,
               'task': 'supervised',
               'loss': myloss,
               'optimizer': 'SGD',
               'init_learning_rate': 0.5
               }

learner = CircuitLearner(hyperparams=hyperparams)

learner.train_circuit(X=X_train, Y=Y_train, steps=100)

test_score = learner.score_circuit(X=X_test, Y=Y_test,
                                   outputs_to_predictions=outputs_to_predictions)
print("\nPossible scores to print: {}".format(list(test_score.keys())))
print("Accuracy on test set: ", test_score['accuracy'])
print("Loss on test set: ", test_score['loss'])

outcomes = learner.run_circuit(X=X_pred, outputs_to_predictions=outputs_to_predictions)

print("\nPossible outcomes to print: {}".format(list(outcomes.keys())))
print("Predictions for new inputs: {}".format(outcomes['predictions']))