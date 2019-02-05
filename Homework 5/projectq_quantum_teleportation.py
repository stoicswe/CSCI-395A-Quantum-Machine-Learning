from projectq.ops import All, CNOT, H, Measure, X, Z
from projectq import MainEngine


def create_bell_pair(quantum_engine):
    # Qubit one is 'Alices' qubit, and will be used to create a message state
    qubit_one = quantum_engine.allocate_qubit()
    # Qubit two is 'Bobs' qubit, and will be used to re-create the message state
    qubit_two = quantum_engine.allocate_qubit()
    ''' 
    Hadamard gate to put Qubit one in superposition
    This sets the value of a equal probability of being 1 or 0
    '''
    H | qubit_one

    '''
    CNOT gate to flip the second Qubit conditonally
    on the first qubit being in the state |1⟩
    '''
    CNOT | (qubit_one, qubit_two)

    return qubit_one, qubit_two


'''
The create_message function takes one of the entangled qubits as a input, 
and a message value. The message value is a bit with the value of 0 or 1. 
The message_value is then entangled with the qubit, and converted into 
two classical bits to send the data through classical channels
'''

def create_message(quantum_engine='', qubit_one='', message_value=0):
    qubit_to_send = quantum_engine.allocate_qubit()
    if message_value == 1:
        '''
        setting the qubit to positive if message_value is 1
        by flipping the base state with a Pauli-X gate.
        '''
        X | qubit_to_send

    # entangle the original qubit with the message qubit
    CNOT | (qubit_to_send, qubit_one)

    '''
    1 - Put the message qubit in superposition 
    2 - Measure out the two values to get the classical bit value
        by collapsing the state. 
    '''
    H | qubit_to_send
    Measure | qubit_to_send
    Measure | qubit_one

    # The qubits are now turned into normal bits we can send through classical channels
    classical_encoded_message = [int(qubit_to_send), int(qubit_one)]

    return classical_encoded_message


'''
The function to receive messages takes the classical encoded
message, along with the second qubit from the Bell pair. 
Then Pauli-X and/or Pauli-Z gates are applied to the Qubit,
conditionally on the values in the message. 
'''

def message_reciever(quantum_engine, message, qubit_two):
    '''
    Pauli-X and/or Pauli-Z gates are applied to the Qubit,
    conditionally on the values in the message.
    '''
    if message[1] == 1:
        X | qubit_two
    if message[0] == 1:
        Z | qubit_two

    '''
    Measuring the Qubit and collapsing the state down to either 1 or 0
    '''
    Measure | qubit_two

    quantum_engine.flush()

    received_bit = int(qubit_two)
    return received_bit


def send_receive(bit=0,quantum_engine=''):
    # Create bell pair
    qubit_one, qubit_two = create_bell_pair(quantum_engine)
    # entangle the bit with the first qubit
    classical_encoded_message = create_message(quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=bit)
    # Teleport the bit and return it back
    return message_reciever(quantum_engine, classical_encoded_message, qubit_two)


def send_full_message(message='DataEspresso.com',quantum_engine=''):
    # Convert the string into binary values
    binary_encoded_message = [bin(ord(x))[2:].zfill(8) for x in message]
    print('Message to send: ', message)
    print('Binary message to send: ', binary_encoded_message)

    '''
    The binary message is divided into an list of each word represented in binary.
    We iterate through each word, and then each bit in the letter.
    Then we append the bits to an list to get back the letter representation
    '''
    received_bytes_list = []
    for letter in binary_encoded_message:
        received_bits = ''
        for bit in letter:
            received_bits = received_bits + str(send_receive(int(bit),quantum_engine))
        received_bytes_list.append(received_bits)

    binary_to_string = ''.join([chr(int(x, 2)) for x in received_bytes_list])
    print('Received Binary message: ', received_bytes_list)
    print('Received message: ', binary_to_string)

# Using the simulator as quantum engine
quantum_engine=MainEngine()
message = 'DataEspresso'
send_full_message(message=message,quantum_engine=quantum_engine )