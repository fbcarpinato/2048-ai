from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
import gym
import numpy as np
from gym import spaces
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class My2048Env(gym.Env):
    def __init__(self):
        super(My2048Env, self).__init__()

        self.options = Options()
        chrome_prefs = {"profile.managed_default_content_settings.images": 2}
        self.options.experimental_options["prefs"] = chrome_prefs
        self.options.add_argument("--blink-settings=imagesEnabled=false")
        self.driver = webdriver.Chrome(ChromeDriverManager().install())

        self.driver.set_page_load_timeout(10)  
        self.driver.get('https://2048.io')
        self.body = self.driver.find_element(By.CSS_SELECTOR, "body")

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=15, shape=(4, 4), dtype=int)

        self.seed_value = 0

    def step(self, action):
        actions = [Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT]
        
        self.body.send_keys(actions[action])

        new_state = self.get_board()
        reward = self.get_score(new_state)

        game_over = self.driver.execute_script('return document.querySelector(".game-over") !== null;')
        done = game_over

        return new_state, reward, done, {}

    def reset(self):
        self.driver.find_element(By.CSS_SELECTOR, '.restart-button').click()
        return self.get_board()

    def render(self, mode='human'):
        pass

    def get_board(self):
        board = [[0]*4 for _ in range(4)]
        tiles = self.driver.execute_script('''
            return Array.from(document.querySelectorAll('.tile')).map(tile => {
                const classList = tile.className.split(' ');
                const positionClass = classList.find(cn => cn.startsWith('tile-position'));
                const [_, __, row, col] = positionClass.split('-');
                const value = parseInt(tile.textContent);
                return [value, row, col];
            });
        ''')

        for value, row, col in tiles:
            row = int(row) - 1
            col = int(col) - 1
            board[row][col] = value

        return board

    def get_score(self, board):
        board = np.array(board)
        reward = sum(sum(cell**2 for cell in row) for row in board)

        return reward

    def seed(self, seed=None):
        np.random.seed(seed)

def main():
    env = My2048Env()
    env = DummyVecEnv([lambda: env])

    hyperparameters = {
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 0.0003,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
        "seed": 0
    }

    model = PPO("MlpPolicy", env, **hyperparameters)

    total_timesteps = 0
    while True:
        model.learn(total_timesteps=10000, reset_num_timesteps=False)
        total_timesteps += 10000
        model.save("2048_agent")

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        print("Total Timesteps:", total_timesteps)
        print("Mean Reward:", mean_reward)

if __name__ == "__main__":
    main()
