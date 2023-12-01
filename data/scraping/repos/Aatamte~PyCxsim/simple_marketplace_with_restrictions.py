import os
import openai

from CAES import Environment, Query, Order, Marketplace
from CAES import Population
from CAES import ActionRestriction
from CAES import OAIAgent


class MyAgent(OAIAgent):
    def __init__(self):
        super(MyAgent, self).__init__()
        self.inventory.set_starting_inventory(
            {"capital": 1000, "socks": 10}
        )

        self.params["max_price"] = 10

    def execute_query(self):
        return Query()


def buy_restriction(agent, order: Order):
    assert order.quantity <= 0


def sell_restriction(agent, order: Order):
    assert order.quantity >= 0


if __name__ == '__main__':
    openai.api_key = os.environ["open_ai_key"]

    env = Environment(visualization=True)

    buyer_restrictions = [
        ActionRestriction(
            action=Order,
            func=buy_restriction,
            message_to_agent_on_trigger="You are a buyer, and cannot sell goods, please have a quantity greater than or equal to 1",
            inform_agent_and_retry_action=True,
            max_retries=1
        )
    ]

    seller_restrictions = [
        ActionRestriction(
            action=Order,
            func=sell_restriction,
            message_to_agent_on_trigger="You are a seller, and cannot buy goods",
            inform_agent_and_retry_action=True
        )
    ]

    buyer_population = Population(agent=MyAgent(), number_of_agents=2, action_restrictions=buyer_restrictions)

    seller_population = Population(agent=MyAgent(), number_of_agents=2, action_restrictions=seller_restrictions)

    env.add(buyer_population)
    env.add(seller_population)

    marketplace = Marketplace()
    env.add(marketplace)

    #dialogue = Dialogue()
    #env.add(dialogue)

    env.step_delay = 2

    env.max_episodes = 1
    env.max_steps = 50

    # set up the environment
    env.set_up()

    for step in env.iter_steps():
        for agent in env.agents:
            print(agent.inventory)
        print(step)
        env.step()

