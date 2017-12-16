from agent_dir.agent import Agent
from model import *

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            print('loading trained model')
            self.net = torch.load('./net_mask_2550000.pt')
            self.net.eval_net.eval()

        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        dqn = DQN(TARGET_UPDATE_FREQ)

        rewards = []
        seed = 11037
        self.env.seed(seed)
        model_path = '/mnt/disk0/kevin1kevin1k/models/'
        steps = 0
        start = time()
        for ep in count(1):
            s = self.env.reset()
            done = False
            episode_reward = 0.0

            for j in count():
                a = dqn.get_action(s, steps / (TOTAL_STEPS * 0.1))
                s_, r, done, info = self.env.step(a)

                dqn.memory.add_transition(
                    FloatTensor(np.expand_dims(s.transpose(2, 0, 1).astype(float), 0)),
                    LongTensor([[a]]),
                    FloatTensor(np.expand_dims(s_.transpose(2, 0, 1).astype(float), 0)) if not done else None,
                    FloatTensor([r.astype(float)]),
                )

                if dqn.can_learn() and steps % EVAL_UPDATE_FREQ == 0:
                    update_target = steps % TARGET_UPDATE_FREQ == 0
                    dqn.learn(update_target)

                episode_reward += r
                steps += 1

                if steps % SAVE_EVERY == 0:
                    torch.save(dqn, MODEL_PATH + 'dqn_test_{}.pt'.format((steps // SAVE_EVERY) % 10))

                if done:
                    break

                s = s_

            rewards.append(episode_reward)
            avg = np.average(rewards[-100:])
            print('Episode: {}, steps: {}, reward: {:.1f}, avg_100: {:.1f}, time: {}'.format(ep, steps, episode_reward, avg, int(time() - start)))
            if steps > TOTAL_STEPS:
                break


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        action = self.net.get_action(observation, -1)
        return action
