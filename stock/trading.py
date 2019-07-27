import math
import pandas_datareader as data_reader
import numpy as np
from stock.ai_trader import AI_Trader
from tqdm import tqdm

class Trading:
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def stocks_price_format(n):
        if n < 0:
            return "- $ {0:2f}".format(abs(n))
        else:
            return "$ {0:2f}".format(abs(n))

    @staticmethod
    def dataset_loader(stock_name):
        dataset = data_reader.DataReader(stock_name, data_source="yahoo")
        start_date = str(dataset.index[0]).split()[0]
        end_date = str(dataset.index[-1]).split()[0]
        close = dataset['Close']
        return close

    def state_creator(self,data, timestep, window_size):
        starting_id = timestep - window_size + 1

        if starting_id >= 0:
            windowed_data = data[starting_id: timestep + 1]
        else:
            windowed_data =- starting_id * [data[0]] + list(data[0:timestep + 1])

        state = []
        for i in range(window_size - 1):
            state.append(self.sigmoid(windowed_data[i + 1] - windowed_data[i]))

        return np.array([state])

    """
    hook method
    """
    def transaction(self, target):
        stock_name = target
        data = self.dataset_loader(stock_name)
        window_size = 10
        episodes = 1000
        batch_size = 32
        data_samples = len(data) - 1
        trader = AI_Trader(window_size)
        print('==== Model Summary ===')
        print(trader.model.summary())
        for episode in range(1, episodes + 1):
            print("Episode: {}/{}".format(episode, episodes))
            state = self.state_creator(data, 0, window_size + 1)
            total_profit = 0
            trader.inventory = []

            for t in tqdm(range(data_samples)):
                action = trader.trade(state)
                next_state = self.state_creator(data, t + 1, window_size + 1)
                reward = 0

                if action == 1: # Buying
                    trader.inventory.append(data[t])
                    print("AI 트레이더 매수: ", self.stocks_price_format(data[t]))
                elif action == 2 and len(trader.inventory) > 0: # Selling
                    buy_price = trader.inventory.pop(0)
                    reward = max(data[t] - buy_price, 0)
                    total_profit += data[t] - buy_price
                    print("AI 트레이더 매도: ", self.stocks_price_format(data[t]),
                          "이익: "+self.stocks_price_format(data[t] - buy_price))
                if t == data_samples - 1:
                    done = True
                else:
                    done = False

                trader.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    print('#################')
                    print('총이익: {}'.format(total_profit))
                    print('#################')

                if len(trader.memory) > batch_size:
                    trader.batch_train(batch_size)
            if episode % 10 == 0:
                trader.model.save('ai_trader_{}.h5'.format(episode))