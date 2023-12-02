import gym
import universe
from ml.reinforcement.transformer.autoencoder_transformer import AutoencoderTransformer
from ml.common.classifier.keras import KerasClassifier
from ml.reinforcement.agent.qlearn_autoencoder import AutoencoderQlearnAgent
from ml.reinforcement.experiment.base import ReinforcementLearningExperiment
from ml.reinforcement.runner.environ.atari_gym_ml import OpenAIAtariGymMLEnvRunner
from keras import Model, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten


class NeonRacerQlearnAutoencoderKerasExperiment(ReinforcementLearningExperiment):
    name = 'neon_racer_qlearn_autoencoder_experiment'

    env_name = 'flashgames.NeonRace-v0'
    env_history_length = 100000
    gamma = 0.95  # discount rate
    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    actions = [
    [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False),
     ('KeyEvent', 'n', False)],
    [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False),
     ('KeyEvent', 'n', False)],
    [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True),
     ('KeyEvent', 'n', False)],
    [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False),
     ('KeyEvent', 'n', True)]
]
    action_shape = (4,)
    action_size = len(actions)
    verbose = True

    height = 60
    width = 60
    num_channels = 3
    observation_shape = (height, width, num_channels)

    def get_model(self):
        input_img = Input(name='input', shape=(self.height, self.width, self.num_channels))
        X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input_img)
        X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(X)

        X = Flatten()(X)
        X = Dense(128, activation='relu')(X)
        y = Dense(self.action_size, name='classifier', activation='softmax')(X)

        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(encoded)
        X = UpSampling2D(size=(2, 2))(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        X = UpSampling2D(size=(2, 2))(X)
        X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(X)
        X = UpSampling2D(size=(2, 2))(X)
        decoded = Conv2D(name='decoder', filters=1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid',
                         padding='same')(X)

        autoencoder = Model(inputs=input_img, outputs=[decoded, y])
        autoencoder.compile(loss={'classifier': 'mse', 'decoder': 'binary_crossentropy'}, optimizer='adam')
        classifier = Model(input=input_img, output=y)
        print(autoencoder.summary())
        print(classifier.summary())
        return autoencoder, classifier

    def get_agent(self):
        autoencoder, classifier = self.get_classifier()

        return AutoencoderQlearnAgent(
            actions=self.actions,
            autoencoder=autoencoder,
            classifier=classifier,
            action_shape=self.action_shape,
            observation_shape=self.observation_shape,
            epsilon=self.epsilon,
            epsilon_decay=self.epsilon_decay,
            epsilon_min=self.epsilon_min,
            gamma=self.gamma,
            verbose=self.verbose
        )

    def get_training_runner(self):
        return OpenAIAtariGymMLEnvRunner(
            env_history_length=self.env_history_length,
            checkpoint_step_num=self.checkpoint_step_num,
            env_name=self.env_name,
            experiment_name=self.name,
            model_dir=self.model_dir,
            agent=self.agent,
            num_epochs=self.num_epochs,
            num_steps=self.num_steps,
            batch_size=self.batch_size
        )

    def get_classifier(self):
        self.transformer = AutoencoderTransformer(width=self.width, height=self.height, num_channels=self.num_channels)
        autoencoder, classifier = self.get_model()

        return KerasClassifier(model=autoencoder, transformer=self.transformer), KerasClassifier(model=classifier, transformer=self.transformer)
