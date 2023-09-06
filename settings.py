__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

# You can manipulate these defaults to change the game parameters.

game_settings = {

    # File implementing the agent playing as player 1
    "player1": "my_agent.py",

    # File implementing the agent playing as player 2
    "player2": "random_agent.py",

    # Size of the game grid (Y,X)
    "gridSize": (11, 11),

    # Number of vacuum cleaners
    "nCleaners": 8,

    # Number of turns per game
    "nTurns": 25,

    # Speed of visualisation ('slow','normal','fast')
    "visSpeed": 'normal',

    # Visualisation resolution
    "visResolution": (1280, 720),

    # Wheter to save final games
    "saveFinalGames": True,

    "seed": 0   # seed for game choices, None for random seed
}
