import pygame
import game_over_screen
import openai
import time
import random

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (139, 69, 19)

BOARD_SIZE = 19
GRID_SIZE = 30
SCREEN_SIZE = (BOARD_SIZE * GRID_SIZE, BOARD_SIZE * GRID_SIZE)

openai.api_key = "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"


class GameBoard:
    def __init__(self, screen, board_size):
        self.screen = screen
        self.board_size = board_size
        self.grid_size = screen.get_width() // (board_size + 2)

    def draw_board(self, stones, black_score, white_score):
        self.screen.fill(BROWN)

        for row in range(self.board_size):
            for col in range(self.board_size):
                x = (col + 1) * self.grid_size
                y = (row + 1) * self.grid_size
                pygame.draw.rect(self.screen, BLACK, (x, y, self.grid_size, self.grid_size), 1)

                stone = stones[row][col]
                if stone == "B":
                    pygame.draw.circle(self.screen, BLACK, (x + self.grid_size // 2, y + self.grid_size // 2),
                                       self.grid_size // 2 - 2)
                elif stone == "W":
                    pygame.draw.circle(self.screen, WHITE, (x + self.grid_size // 2, y + self.grid_size // 2),
                                       self.grid_size // 2 - 2)

        # Draw black score
        black_text = f"Black: {black_score}"
        black_font = pygame.font.Font(None, 18)
        black_text_render = black_font.render(black_text, True, BLACK)
        black_text_rect = black_text_render.get_rect(left=self.grid_size,
                                                     bottom=self.screen.get_height() - self.grid_size // 2)
        self.screen.blit(black_text_render, black_text_rect)

        # Draw white score
        white_text = f"White: {white_score}"
        white_font = pygame.font.Font(None, 18)
        white_text_render = white_font.render(white_text, True, WHITE)
        white_text_rect = white_text_render.get_rect(right=self.screen.get_width() - self.grid_size,
                                                     bottom=self.screen.get_height() - self.grid_size // 2)
        self.screen.blit(white_text_render, white_text_rect)

        pygame.display.flip()


def generate_input_text(stones, current_player):
    # Generate the input text for ChatGPT based on the current board state
    input_text = "Go game board:\n"
    for row in stones:
        input_text += "".join(row) + "\n"
    input_text += "Current player: " + current_player + "\n"
    input_text += "Your move: "

    return input_text


def convert_ai_move(ai_move):
    # Convert AI's move string to board coordinates (x, y)
    x = ord(ai_move[0].lower()) - ord('a')
    y = int(ai_move[1:]) - 1
    return x, y


def handle_player_move(stones, current_player, game_board):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                grid_x = (mouse_pos[0] - game_board.grid_size) // game_board.grid_size
                grid_y = (mouse_pos[1] - game_board.grid_size) // game_board.grid_size

                if (
                        grid_x >= 0
                        and grid_x < BOARD_SIZE
                        and grid_y >= 0
                        and grid_y < BOARD_SIZE
                        and stones[grid_y][grid_x] == ""
                ):
                    stones[grid_y][grid_x] = current_player
                    return grid_x, grid_y


def handle_ai_move(stones, current_player):
    # AI player's turn
    start_time = time.time()
    while True:
        if time.time() - start_time > 5:  # AI has 5 seconds to make a move
            # AI took too long, make a random move
            valid_moves = [(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if stones[y][x] == ""]
            if valid_moves:
                x, y = random.choice(valid_moves)
                stones[y][x] = current_player
                return x, y

        # Generate AI move using ChatGPT
        input_text = generate_input_text(stones, current_player)
        response = openai.Completion.create(
            engine='davinci',
            prompt=input_text,
            max_tokens=1,
            n=1,
            stop=None,
            temperature=0.6,
        )
        ai_move = response.choices[0].text.strip()

        # Convert AI move to board coordinates
        if ai_move:
            x, y = convert_ai_move(ai_move)
            if (
                    x >= 0
                    and x < BOARD_SIZE
                    and y >= 0
                    and y < BOARD_SIZE
                    and stones[y][x] == ""
            ):
                stones[y][x] = current_player
                return x, y


def is_valid_coordinate(x, y):
    return x >= 0 and x < BOARD_SIZE and y >= 0 and y < BOARD_SIZE


def get_neighbors(x, y):
    neighbors = [
        (x - 1, y),
        (x + 1, y),
        (x, y - 1),
        (x, y + 1)
    ]
    return [(x, y) for x, y in neighbors if is_valid_coordinate(x, y)]


def remove_captured_stones(stones, x, y, current_player):
    captured_stones = []
    neighbors = get_neighbors(x, y)

    for nx, ny in neighbors:
        if stones[ny][nx] == current_player:
            continue

        group, liberties = find_group(nx, ny, stones, [])
        if liberties == 0:
            captured_stones.extend(group)

    for cx, cy in captured_stones:
        stones[cy][cx] = ""

    return len(captured_stones)


def find_group(x, y, stones, visited):
    visited.append((x, y))
    group = [(x, y)]
    liberties = 0

    for nx, ny in get_neighbors(x, y):
        if (nx, ny) in visited:
            continue

        if stones[ny][nx] == "":
            liberties += 1
        elif stones[ny][nx] == stones[y][x]:
            g, l = find_group(nx, ny, stones, visited)
            group.extend(g)
            liberties += l

    return group, liberties


def count_territory(stones):
    territory = {"B": 0, "W": 0}

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if stones[row][col] == "":
                group, color = find_territory(row, col, stones)
                territory[color] += len(group)

    return territory


def find_territory(x, y, stones):
    visited = []
    group = []
    color = ""

    if (x, y) in visited:
        return group, color

    if stones[y][x] == "":
        color = "B"
    else:
        color = stones[y][x]

    queue = [(x, y)]
    while queue:
        cx, cy = queue.pop(0)
        if (cx, cy) in visited:
            continue

        visited.append((cx, cy))
        group.append((cx, cy))

        for nx, ny in get_neighbors(cx, cy):
            if stones[ny][nx] == "":
                color = ""
            elif stones[ny][nx] != color:
                color = "B"

            if (nx, ny) not in visited:
                queue.append((nx, ny))

    return group, color or "B"


def play_game(screen, selected_player):
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Go Game")
    game_board = GameBoard(screen, BOARD_SIZE)

    stones = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = "B"

    black_stones_limit = 181  # Set black stones limit to 181
    white_stones_limit = 180  # Set white stones limit to 180
    black_prisoners = 0
    white_prisoners = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game_board.draw_board(stones, black_prisoners, white_prisoners)

        if current_player == selected_player:
            x, y = handle_player_move(stones, current_player, game_board)
        else:
            x, y = handle_ai_move(stones, current_player)

        if x is not None and y is not None:
            remove_captured_stones(stones, x, y, current_player)

        # Check for captured stones and update statistics
        black_captured = 0
        white_captured = 0

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if stones[i][j] == "":
                    neighbors = get_neighbors(i, j)
                    neighbor_colors = [stones[x][y] for x, y in neighbors if is_valid_coordinate(x, y)]
                    if "B" in neighbor_colors and "W" not in neighbor_colors:
                        black_captured += 1
                    elif "W" in neighbor_colors and "B" not in neighbor_colors:
                        white_captured += 1

        black_prisoners += black_captured
        white_prisoners += white_captured

        # Check if any player reached the stone limit
        if black_stones_limit <= 0 and white_stones_limit <= 0:
            if black_prisoners < white_prisoners:
                game_over_screen("B")  # Show game over screen with Black player as winner
            elif white_prisoners < black_prisoners:
                game_over_screen("W")  # Show game over screen with White player as winner
            else:
                game_over_screen("")  # Show game over screen as a tie

            running = False

        # Switch players
        current_player = "W" if current_player == "B" else "B"

        pygame.display.flip()

    pygame.quit()


def main():
    pygame.init()

    screen_size = (400, 300)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Go Game")

    # Player selection menu
    selected_player = None
    while selected_player is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if 100 <= mouse_pos[0] <= 300 and 100 <= mouse_pos[1] <= 200:
                    if mouse_pos[0] < 200:
                        selected_player = "B"  # Human plays as Black
                    else:
                        selected_player = "W"  # Human plays as White

        screen.fill((255, 255, 255))  # White

        # Draw player selection buttons
        pygame.draw.rect(screen, BLACK, (100, 100, 100, 100))  # Black player button
        pygame.draw.rect(screen, WHITE, (200, 100, 100, 100))  # White player button

        pygame.display.flip()

    play_game(screen, selected_player)  # Pass the selected player to the play_game function

    pygame.quit()


if __name__ == "__main__":
    main()
