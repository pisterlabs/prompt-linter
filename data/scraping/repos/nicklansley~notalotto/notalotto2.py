# Simpler answer from OpenAI's ChatGPT service
import random

# Euromillions lottery parameters
NUM_BALLS = 50
NUM_STARS = 12


def simulate_draw():
    # Simulate a draw by selecting 5 balls and 2 stars at random
    balls = random.sample(range(1, NUM_BALLS + 1), 5)
    stars = random.sample(range(1, NUM_STARS + 1), 2)
    return balls, stars


def simulate_ticket():
    # Simulate a Euromillions ticket by selecting 5 balls and 2 stars at random
    balls = sorted(random.sample(range(1, NUM_BALLS + 1), 5))
    stars = sorted(random.sample(range(1, NUM_STARS + 1), 2))
    return balls, stars


def compare_tickets(ticket, draw):
    # Compare a ticket to a draw and return the number of balls and stars that match
    balls_match = len(set(ticket[0]).intersection(set(draw[0])))
    stars_match = len(set(ticket[1]).intersection(set(draw[1])))
    return balls_match, stars_match


def simulate_lottery(number_of_draws):
    # Simulate a series of Euromillions draws and return the number of times that a ticket wins
    win_count = 0
    for _ in range(number_of_draws):
        draw = simulate_draw()
        ticket = simulate_ticket()
        matches = compare_tickets(ticket, draw)
        if matches[0] == 5 and matches[1] == 2:
            win_count += 1
    return win_count


if __name__ == '__main__':
    # Run the simulation
    num_draws = 10000000
    wins = simulate_lottery(num_draws)

    # Print the results
    print(f"After {num_draws} draws, a ticket matched all 5 balls and 2 stars {wins} times.")
    if wins > 0:
        print(f"The probability of winning the Euromillions lottery is 1 in {num_draws / wins:.0f}.")
    else:
        print("The probability of winning the Euromillions lottery is 0.")
