# Tic Tac Toe - Terminal Based Game

This is a simple terminal-based Tic Tac Toe game implemented in Python. The game allows two players to play Tic Tac Toe in the terminal.

## How to Play

1. The game board is a 3x3 grid.
2. Players take turns to place their mark (X or O) on the board.
3. The first player to get three of their marks in a row (horizontally, vertically, or diagonally) wins the game.
4. If all nine cells are filled and no player has three marks in a row, the game is a draw.

## Getting Started

### Prerequisites

- Python 3.x

### Running the Game

1. Clone the repository or download the `tic_tac_toe.py` file.
2. Open a terminal and navigate to the directory containing `tic_tac_toe.py`.
3. Run the following command to start the game:

    ```sh
    python tic_tac_toe.py
    ```

4. Follow the on-screen instructions to play the game.

## Game Instructions

- Players will be prompted to enter their move.
- Enter a number between 1 and 9 to place your mark on the board.
- The numbers correspond to the cells on the board as follows:

    ```
     1 | 2 | 3
    -----------
     4 | 5 | 6
    -----------
     7 | 8 | 9
    ```

- The game will display the updated board after each move.
- The game will announce the winner or if the game is a draw.

## Example

```
Player X, enter your move (1-9): 1
 X |   |  
-----------
   |   |  
-----------
   |   |  
Player O, enter your move (1-9): 5
 X |   |  
-----------
   | O |  
-----------
   |   |  
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
