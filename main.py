import keyboard
import time
import numpy as np
import math
from PIL import ImageGrab
from TetrisBoard import TetrisBoard
import pyautogui # somehow the mouse get_position doesn't work

x1_board, y1_board = 823, 251 # top left of board
x2_board, y2_board = 1145, 889 # bottom right of board
x1, y1 = 1250, 326
x2, y2 = 0, 0
x3, y3 = 0, 0
x4, y4 = 0, 0
x5, y5 = 1243, 736


pixel_area = 30 # number of pixels to check for color - auto

# keybinds
rotate_clockwise_key = 'x' # for some reason this is the only key that works - 'up' doesn't work
rotate_180_key = 'a'
rotate_counterclockwise_key = 'z'
move_left_key = 'left'
move_right_key = 'right'
drop_key = 'space'
# constants - ARR 0ms - DAS 40ms
calculation_accuracy = 8 # number of best moves to keep at each depth - higher number means more accurate but slower
max_depth = 6 # number of moves into the future to simulate, max is 6, you can only see 6 blocks at once - higher number means more accurate but slower
wait_time = 0.04 # time to wait, can't go too low because you need to wait for screen to refresh
scan_board = True # some modes require scanning the board because of extra pieces - zen, multiplayer
jstris = False # jstris mode - changes colors

# Game Settings - DAS 40ms, ARR 0ms

key_delay = 0

# Colors for tetrio
colors = [
    (194, 64, 70),  # red - Z
    (142, 191, 61),  # lime - Z2
    (93, 76, 176), # dark blue - L2
    (192, 168, 64),  # yellow - O
    (62, 191, 144),  # turquoise - I
    (194, 115, 68), # orange - L
    (176, 75, 166), # purple - T
]

# jstris settings
if jstris:
    # keybinds
    rotate_clockwise_key = 'up'
    rotate_180_key = 'a'
    rotate_counterclockwise_key = 'z'
    move_left_key = 'left'
    move_right_key = 'right'
    drop_key = 'space'
    # colors for jstris
    colors = [
        (215, 15, 55),  # red - Z
        (89, 177, 1),  # lime - Z2
        (33, 65, 198), # dark blue - L2
        (227, 159, 2),  # yellow - O
        (15, 155, 215),  # turquoise - I
        (227, 91, 2), # orange - L
        (175, 41, 138), # purple - T
    ]

    x1_board, y1_board = 1100,228 # top left of board
    x2_board, y2_board = 1399,827 # bottom right of board
    x1, y1 = 1475,289
    x2, y2 = 0, 0
    x3, y3 = 0, 0
    x4, y4 = 0, 0
    x5, y5 = 1475,649

# Each piece is represented by a 2D array, and rotations are stored as a list of 2D arrays
# 4x4 pieces are padded with 0s to make them 4x4
tetris_pieces = {
    'I': [
        np.array([[1, 1, 1, 1]]),
        np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]])
    ],
    'O': [
        np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
    ],
    'T': [
        np.array([[1, 1, 1, 0], [0, 1, 0, 0]]),
        np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]]),
        np.array([[0, 1, 0, 0], [1, 1, 1, 0]]),
        np.array([[0, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]]),
    ],
    'L': [
        np.array([[1, 1, 1, 0], [0, 0, 1, 0]]),
        np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),
        np.array([[1, 0, 0, 0], [1, 1, 1, 0]]),
        np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]]),
    ],
    'L2': [
        np.array([[1, 1, 1, 0], [1, 0, 0, 0]]),
        np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]]),
        np.array([[0, 0, 1, 0], [1, 1, 1, 0]]),
        np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]),
    ],
    'Z': [
        np.array([[0, 1, 1, 0], [1, 1, 0, 0]]),
        np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]])
    ],
    'Z2': [
        np.array([[1, 1, 0, 0], [0, 1, 1, 0]]),
        np.array([[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]])
    ]
}


def evaluate_board(board):
    # Implement your heuristic function here
    # The height of the tallest column - find highest row with a 1
    highest_block_row = 20
    for row in range(board.shape[0]):
        if not np.any(board[row] == 1):
            highest_block_row = row
            break
    # The sum of max height block in each column - the bottom of the board is index 0
    sum_of_heights = 0
    for col in range(board.shape[1]):
        for row in reversed(range(board.shape[0])):
            if board[row][col] == 1:
                sum_of_heights += row + 1
                break
    num_cleared_rows = np.sum(np.all(board == 1, axis=1))
    # The number of holes - find number of 0s with 1s above
    holes = np.sum((board == 0) & (np.cumsum(board, axis=0) < np.sum(board, axis=0)))
    # The number of blockades - find number of 1s with 0s above
    blockades = np.sum((board == 1) & (np.cumsum(board, axis=0) > 0))
    # assign higher weights to higher blocks
    weighted_heights = 0
    for col in range(board.shape[1]):
        for row in reversed(range(board.shape[0])):
            # find highest block in each column
            if board[row][col] == 1:
                if row > 5:
                    weighted_heights += (row + 1) * (row + 1 - 5)
                else:
                    weighted_heights += (row + 1)
                break

    A, B, C, D, E = -1, 10, -50, -1, -1
    score = A * weighted_heights + B * num_cleared_rows * num_cleared_rows * num_cleared_rows + C * holes + D * blockades + E * highest_block_row
    return score

def get_positions(board, rotated_block):
    # Return a list of all possible positions for the given block and rotation
    possible_positions = []
    # remove padded 0s from rotated block
    rotated_block = rotated_block[~np.all(rotated_block == 0, axis=1)]
    rotated_block = rotated_block[:, ~np.all(rotated_block == 0, axis=0)]
    # drop block from top for each column - bottom is index 0
    for x in range(board.shape[1] - rotated_block.shape[1] + 1):
        y = board.shape[0] - rotated_block.shape[0] - 1
        while y >= 0:
            if np.any(np.logical_and(rotated_block, board[y:y + rotated_block.shape[0], x:x + rotated_block.shape[1]])):
                if y > board.shape[0] - rotated_block.shape[0]:
                    print("You lose!")
                    break
                    # exit()
                possible_positions.append((y + 1, x))
                break
            if y == 0:
                possible_positions.append((y, x))
            y -= 1

    return possible_positions

def clear_full_rows(board):
    while True:
        for y, row in enumerate(board):
            if all(cell == 1 for cell in row):
                board = np.delete(board, y, axis=0)
                # insert new row at last index which is the top of the board
                board = np.insert(board, board.shape[0], 0, axis=0)
                break
            if y == board.shape[0] - 1:
                return board
            
def num_of_full_rows(board):
    return np.sum(np.all(board == 1, axis=1))

def find_least_holes(board):
    return np.sum((board == 0) & (np.cumsum(board, axis=0) < np.sum(board, axis=0)))



def find_best_position(board, block_array, depth):
    def helper(boards, block, position_rotations_array, num_boards_keep):
        best_ret = None
        new_boards = []
        new_position_rotations = []
        score_list = []

        for b_idx, b in enumerate(boards):
            for rotation in range(len(block)):
                positions = get_positions(b, block[rotation])
                if not positions:
                    continue
                for pos in positions:
                    new_board = place_block(b, block[rotation], pos)
                    score = evaluate_board(new_board)
                    score_list.append(score)
                    new_boards.append(new_board)
                    if position_rotations_array is None:
                        new_position_rotations.append((pos, rotation))
                    else:
                        new_position_rotations.append(position_rotations_array[b_idx])

        if not score_list:
            return None  # aucun placement valide

        # récupère le meilleur
        best_index = max(range(len(score_list)), key=lambda i: score_list[i])
        return new_position_rotations[best_index]

    # début logique principale
    if not block_array:
        return (5, 0)

    if depth < 1:
        depth = 1

    # ajoute des ghost pieces si besoin
    while len(block_array) < depth:
        piece = tetris_pieces[list(tetris_pieces.keys())[np.random.randint(len(tetris_pieces))]]
        block_array.append(piece)

    current_boards = [board.copy()]
    current_positions = [None]

    for i in range(depth):
        block = block_array[i]
        ret = helper(current_boards, block, None if i == 0 else current_positions, calculation_accuracy if i < depth - 1 else 1)
        if ret is None:
            return (5, 0)  # fallback safe
        pos, rotation = ret
        # simuler la pose dans le board pour la suite
        # optional: update current_boards accordingly
        return (pos, rotation)

    # fallback général
    return (5, 0)




def place_block(board, rotated_block, position):
    new_board = board.copy()
    # remove padded 0s from rotated block
    rotated_block = rotated_block[~np.all(rotated_block == 0, axis=1)]
    rotated_block = rotated_block[:, ~np.all(rotated_block == 0, axis=0)]
    new_board[position[0]:position[0] + rotated_block.shape[0], position[1]:position[1] + rotated_block.shape[1]] += rotated_block
    return new_board

def euclidean_distance(color1, color2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

def closest_color_in_area(colors, x, y):
    min_diff = float('inf')
    closest_color = (0, 0, 0)
    while min_diff > 20:
        # break if esc
        if keyboard.is_pressed('esc'):
            break
        # get pixel colors in a 10 by 10 around x and y
        target_colors = []
        # Grab a portion of the screen
        half = pixel_area//2
        full = half * 2
        image = ImageGrab.grab(bbox=(x - half, y - half, x + half, y + half))
        # Loop through the pixels in the grabbed image
        for i in range(full):
            for j in range(full):
                target_colors.append(image.getpixel((i, j)))

        # find the closest color in target_colors that is in colors
        closest_color = (0, 0, 0)
        min_diff = float('inf')
        for target_color in target_colors:
            for color in colors:
                diff = euclidean_distance(color, target_color)
                if diff < min_diff:
                    min_diff = diff
                    closest_color = color
                if min_diff < 20:
                    break
    return tuple(closest_color)

def get_piece_based_on_color(matched_color, colors):
    piece = None
    if matched_color == colors[0]:
        # print('Red - Z')
        piece = tetris_pieces['Z']
    elif matched_color == colors[1]:
        # print('Lime - Z2')
        piece = tetris_pieces['Z2']
    elif matched_color == colors[2]:
        # print('Dark blue - L2')
        piece = tetris_pieces['L2']
    elif matched_color == colors[3]:
        # print('Yellow - O')
        piece = tetris_pieces['O']
    elif matched_color == colors[4]:
        # print('Turquoise - I')
        piece = tetris_pieces['I']
    elif matched_color == colors[5]:
        # print('Orange - L')
        piece = tetris_pieces['L']
    elif matched_color == colors[6]:
        # print('Purple - T')
        piece = tetris_pieces['T']
    if piece is None:
        print('No piece found')
    return piece


# Create a new board
tetrisboard = TetrisBoard()
board_initialized = False
piece_array = []

def key_press(best_position, best_rotation, current_x=3):
    import random

    def human_delay():
        time.sleep(random.uniform(0.03, 0.05))  # 15 à 50 ms entre actions

    # Rotation
    if best_rotation == 1:
        keyboard.press_and_release(rotate_clockwise_key)
        human_delay()
    elif best_rotation == 2:
        keyboard.press_and_release(rotate_180_key)
        human_delay()
    elif best_rotation == 3:
        keyboard.press_and_release(rotate_counterclockwise_key)
        human_delay()

    # Déplacement horizontal
    for _ in range(abs(best_position[1] - current_x)):
        if best_position[1] < current_x:
            keyboard.press_and_release(move_left_key)
        else:
            keyboard.press_and_release(move_right_key)
        human_delay()

    # Hard drop
    keyboard.press_and_release(drop_key)
    human_delay()




def get_tetris_board_from_screen(top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    board_coords = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    board_image = ImageGrab.grab(board_coords)
    board_image = board_image.convert('L')
    board = np.zeros((20, 10), dtype=int)
    block_width = board_image.width / 10
    block_height = board_image.height / 20

    for row in reversed(range(20)):
        empty_row = True
        for col in range(10):
            total_darkness = 0
            num_pixels = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x = math.floor(col * block_width + block_width / 2) + dx
                    y = math.floor(row * block_height + block_height / 2) + dy
                    pixel_value = board_image.getpixel((x, y))
                    total_darkness += pixel_value
                    num_pixels += 1
            avg_darkness = total_darkness / num_pixels

            if avg_darkness < 30:
                board[20 - row - 1][col] = 0
            else:
                empty_row = False
                board[20 - row - 1][col] = 1
        time.sleep(0.015)
        if empty_row:
            break
    return board



# start program
while True:
    if keyboard.is_pressed('1'):
        x1, y1 = pyautogui.position()
        print(f"first piece: {x1},{y1}")
        time.sleep(0.2)

    if keyboard.is_pressed('2'):
        x5, y5 = pyautogui.position()
        print(f"fifth piece: {x5},{y5}")
        time.sleep(0.2)
    
    if keyboard.is_pressed('3'):
        x1_board, y1_board = pyautogui.position()
        print(f"top left: {x1_board},{y1_board}")
        time.sleep(0.2)

    if keyboard.is_pressed('4'):
        x2_board, y2_board = pyautogui.position()
        print(f"bottom right: {x2_board},{y2_board}")
        time.sleep(0.2)

    # Exit the loop with the "ESC" key
    if keyboard.is_pressed('esc'):
        break

    if x1 != 0 and x5 != 0 and not board_initialized and keyboard.is_pressed('space'):
        print('Board initialized')
        board_initialized = True
        # calculate pixel_area
        pixel_area = (y5 - y1)//10
        print("pixel_area: ", pixel_area)
        x2, y2 = (x1+x5)//2, y1+math.floor(((y5-y1)/4)*1)
        x3, y3 = (x1+x5)//2, y1+math.floor(((y5-y1)/4)*2)
        x4, y4 = (x1+x5)//2, y1+math.floor(((y5-y1)/4)*3)
        closest_color1 = closest_color_in_area(colors, x1, y1)
        closest_color2 = closest_color_in_area(colors, x2, y2)
        closest_color3 = closest_color_in_area(colors, x3, y3)
        closest_color4 = closest_color_in_area(colors, x4, y4)
        closest_color5 = closest_color_in_area(colors, x5, y5)
        piece_array.append(get_piece_based_on_color(closest_color1, colors))
        piece_array.append(get_piece_based_on_color(closest_color2, colors))
        piece_array.append(get_piece_based_on_color(closest_color3, colors))
        piece_array.append(get_piece_based_on_color(closest_color4, colors))
        piece_array.append(get_piece_based_on_color(closest_color5, colors))
        print(f'Closest color: {closest_color1}')
        print(f'Closest color: {closest_color2}')
        print(f'Closest color: {closest_color3}')
        print(f'Closest color: {closest_color4}')
        print(f'Closest color: {closest_color5}')
        first_move = True
        while True:
            # set break key
            if keyboard.is_pressed('esc'):
                break
            # restart
            if keyboard.is_pressed(';'):
                board_initialized = False
                break
            # lock until piece changes
            if first_move:
                closest_color1_0 = closest_color_in_area(colors, x1, y1)
                closest_color2_0 = closest_color_in_area(colors, x2, y2)
                if closest_color2 != closest_color2_0 or closest_color1 != closest_color1_0:
                    first_move = False
                else:
                    continue
            # total time
            start_time = time.time()
            # time to get color
            start_time3 = time.time()
            closest_color5 = closest_color_in_area(colors, x5, y5)
            piece_array.append(get_piece_based_on_color(closest_color5, colors))
            print("time for get color: ", time.time() - start_time)
            if scan_board:
                # time to get board
                start_time2 = time.time()
                # get board from screen
                tetrisboard.board = get_tetris_board_from_screen(x1_board, y1_board, x2_board, y2_board)
                for row in reversed(tetrisboard.board):
                    print(row)
                print("time for get board: ", time.time() - start_time2)
            # time how long find_best_position takes
            start_time2 = time.time()
            best_position, best_rotation = find_best_position(tetrisboard.board, piece_array.copy(), max_depth)
            print("time for find_best_position: ", time.time() - start_time2)

            # --- protection : si aucune position trouvée ---
            if best_position is None or best_rotation is None:
                print("Aucune position valide trouvée → sécurité anti-spam activée")
                time.sleep(0.05)
                continue

            # --- protection : si le board devient trop haut ---
            max_height = max((i for i, row in enumerate(tetrisboard.board) if any(row)), default=0)
            if max_height > 17:
                print("PANIC MODE ACTIVÉ — jeu simplifié")
                # profondeur 1 = beaucoup plus stable
                best_position, best_rotation = find_best_position(tetrisboard.board, piece_array.copy(), 1)
            best_piece_pos_rot = piece_array[0][best_rotation]
            # remove first piece from piece_array
            piece_array.pop(0)
            # add offset depending on padded zeros on the left side of axis 1 only
            offset = 0
            for i in range(best_piece_pos_rot.shape[1]):
                if not any(best_piece_pos_rot[:, i]):
                    offset += 1
                else:
                    break
            best_position2 = (best_position[0], best_position[1] - offset)
            # key presses time
            start_time4 = time.time()
            key_press(best_position2, best_rotation)
            print("time for key presses: ", time.time() - start_time4)
            # remove 0s padding
            best_piece_pos_rot = best_piece_pos_rot[~np.all(best_piece_pos_rot == 0, axis=1)]
            best_piece_pos_rot = best_piece_pos_rot[:, ~np.all(best_piece_pos_rot == 0, axis=0)]
            tetrisboard.add_piece(best_piece_pos_rot, best_position)
            # clear full rows
            tetrisboard.clear_full_rows()
            time.sleep(wait_time) # this is needed for some reason (maybe wait for screen to refresh), probably can find a better way
            print("total time: ", time.time() - start_time)
            elapsed = time.time() - start_time
            min_loop = 0.08  # 50ms minimum par boucle
            if elapsed < min_loop:
                time.sleep(min_loop - elapsed)
def closest_color_in_area(colors, x, y):
    min_diff = float('inf')
    closest_color = (0, 0, 0)
    while min_diff > 20:
        if keyboard.is_pressed('esc'):
            break
        # utiliser une zone moyenne de pixels autour de (x, y)
        half = pixel_area // 2
        image = ImageGrab.grab(bbox=(x - half, y - half, x + half, y + half))
        pixels = np.array(image)
        avg_color = tuple(np.mean(pixels.reshape(-1, 3), axis=0).astype(int))

        # comparer avec les couleurs connues
        for color in colors:
            diff = euclidean_distance(color, avg_color)
            if diff < min_diff:
                min_diff = diff
                closest_color = color
        if min_diff < 20:
            break
    return closest_color

