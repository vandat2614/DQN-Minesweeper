import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import Optional, Tuple
from pathlib import Path

MINE = -1
CLOSED = -2
EXPLODED = -3


def is_valid(row: int, col: int, height: int, width: int) -> bool:
    return 0 <= row < height and 0 <= col < width


def is_new_move(board: np.ndarray, row: int, col: int) -> bool:
    return board[row, col] == CLOSED


def is_win(visible_board: np.ndarray, num_mines: int) -> bool:
    return np.count_nonzero(visible_board == CLOSED) == num_mines


def place_mines(height: int, width: int, num_mines: int, forbidden: Tuple[int, int]) -> np.ndarray:
    board = np.zeros((height, width), dtype=int)
    all_indices = [(r, c) for r in range(height) for c in range(width) if (r, c) != forbidden]
    mine_indices = np.random.choice(len(all_indices), size=num_mines, replace=False)
    for idx in mine_indices:
        r, c = all_indices[idx]
        board[r, c] = MINE
    return board


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, width=10, height=10, num_mines=10, action_type='multi', render_mode=None):
        super().__init__()
        assert action_type in ["multi", "discrete"]
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.action_type = action_type
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=-3, high=9, shape=(self.height, self.width), dtype=np.int32)
        if action_type == "multi":
            self.action_space = spaces.MultiDiscrete([self.height, self.width])
        else:
            self.action_space = spaces.Discrete(self.height * self.width)

        self._init_gui_vars()
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.hidden_board = None
        self.playerfield = np.full((self.height, self.width), CLOSED, dtype=int)
        if self.action_type == "multi":
            self.valid_actions = np.ones((self.height, self.width), dtype=bool)
        else:
            self.valid_actions = np.ones(self.height * self.width, dtype=bool)
        self.first_step = True

        if self.render_mode == "human":
            self._init_gui()

        return self.playerfield, {}

    def count_mines_around(self, row: int, col: int) -> int:
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if (dr == 0 and dc == 0) or not is_valid(nr, nc, self.height, self.width):
                    continue
                if self.hidden_board[nr, nc] == MINE:
                    count += 1
        return count

    def open_neighbours(self, row: int, col: int):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if is_valid(nr, nc, self.height, self.width) and is_new_move(self.playerfield, nr, nc):
                    self.playerfield[nr, nc] = self.count_mines_around(nr, nc)
                    if self.playerfield[nr, nc] == 0:
                        self.open_neighbours(nr, nc)

    def step(self, action):
        if self.action_type == "multi":
            row, col = int(action[0]), int(action[1])
        else:
            row, col = divmod(int(action), self.width)

        if not is_new_move(self.playerfield, row, col):
            info = { "is_win": is_win(self.playerfield, self.num_mines)}
            self.valid_actions = (self.playerfield == CLOSED)
            if self.action_type == 'discrete':
                self.valid_actions = self.valid_actions.flatten()
            return self.playerfield, 0, False, False, info

        if self.first_step:
            self.hidden_board = place_mines(
                self.height, self.width, self.num_mines, forbidden=(row, col)
            )
            self.first_step = False

        done = False
        reward = 0

        if self.hidden_board[row, col] == MINE:
            self.playerfield[row, col] = EXPLODED
            done = True
            reward = -1
        else:
            self.playerfield[row, col] = self.count_mines_around(row, col)
            if self.playerfield[row, col] == 0:
                self.open_neighbours(row, col)

        win = is_win(self.playerfield, self.num_mines)
        if not done and win:
            reward = 1
            done = True
        elif not done:
            reward = 0.1

        info = {
            "is_win": win
        }

        if self.render_mode == "human":
            self.render()

        self.valid_actions = (self.playerfield == CLOSED)
        if self.action_type == 'discrete':
            self.valid_actions = self.valid_actions.flatten()

        return self.playerfield, reward, done, False, info


    # GUI INIT AND RENDER
    def _init_gui_vars(self):
        self.tile_size = 32
        self.img_folder = Path(__file__).parent / "img"
        self.tile_dict = {}

    def _init_gui(self):
        pygame.init()
        pygame.mixer.quit()
        screen_width = self.width * self.tile_size
        screen_height = self.height * self.tile_size + 0
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Minesweeper")
        self.clock = pygame.time.Clock()
        self._load_tiles()

    def _load_tiles(self):
        self.tile_dict = {
            -3: pygame.image.load(str(self.img_folder / "explode.jpg")).convert(),
            -1: pygame.image.load(str(self.img_folder / "mine.jpg")).convert(),
            -2: pygame.image.load(str(self.img_folder / "hidden.jpg")).convert(),
            0: pygame.image.load(str(self.img_folder / "0.jpg")).convert(),
            1: pygame.image.load(str(self.img_folder / "1.jpg")).convert(),
            2: pygame.image.load(str(self.img_folder / "2.jpg")).convert(),
            3: pygame.image.load(str(self.img_folder / "3.jpg")).convert(),
            4: pygame.image.load(str(self.img_folder / "4.jpg")).convert(),
            5: pygame.image.load(str(self.img_folder / "5.jpg")).convert(),
            6: pygame.image.load(str(self.img_folder / "6.jpg")).convert(),
            7: pygame.image.load(str(self.img_folder / "7.jpg")).convert(),
            8: pygame.image.load(str(self.img_folder / "8.jpg")).convert(),
        }

    def render(self):
        if self.render_mode != "human":
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.screen.fill((0, 0, 0))
        for r in range(self.height):
            for c in range(self.width):
                val = self.playerfield[r, c]
                tile = self.tile_dict.get(val, self.tile_dict[-1])
                self.screen.blit(tile, (c * self.tile_size, r * self.tile_size))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
