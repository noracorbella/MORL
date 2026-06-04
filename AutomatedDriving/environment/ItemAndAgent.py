import numpy as np
import constants

# un altre intent de veure si els canvis del github i tal i qual funcionen correctament!

def compare_positions(pos1, pos2):
    if pos1[0] == pos2[0] and pos1[1] == pos2[1]:
        return True

    return False


class Move:
    def __init__(self, mover, moved, origin, destination):
        """
        Moves have a mover agent ID, a moved item ID, an origin and a destination.
        The moved item is NOT declared here but in another method.
        :param mover:  the Agent ID (int) that will do the move
        :param origin: The original location of the moved item.
        :param destination: The destination of the moved item.
        """
        self.mover = mover
        self.moved = moved
        self.origin = origin
        self.destination = destination

        self.fast = False
        self.in_between = [-9, -9]

        if np.abs(origin[0]-destination[0]) == 2 or np.abs(origin[1]-destination[1]) == 2:
            self.fast = True
            self.in_between = [int((origin[0]+destination[0])/2), int((origin[1]+destination[1])/2)]

    def get_mover(self):
        """
        :return: the mover agent ID
        """
        return self.mover

    def get_moved(self):
        """

        :return: the moved item ID.
        """
        return self.moved

    def get_origin(self):
        """

        :return: the moved item location
        """
        return self.origin

    def get_destination(self):
        """
        :return: the moved item desired destination.
        """
        return self.destination

    def get_in_between(self):

        return self.in_between

    def check_validity(self, mat):
        """
        Simple method to avoid code problems later. Simply checks if the destination coordinates actually exists.
        Doesn't bother checking if the destination is an unreachable cell.
        :param mat: the matrix representing the map
        :return: true or false according to the analysis
        """

        limits = mat.shape

        for i in range(len(self.destination)):
            if self.destination[i] < 0 or self.destination[i] > (limits[i] - 1):
                return False

        return True


class Item:

    def __init__(self, name, pos):
        """
        Items are very basic classes. They simply have a name/ID (integer) and a position that may change.
        :param name: an integer (be wary of not repeating it, very important).
        :param pos: the position, two integers.
        """
        self.name = name
        self.position = pos
        self.previous_position = pos
        self.origin = pos

    def move(self, pos):
        self.previous_position = self.position
        self.position = pos

    def get_position(self):
        return self.position

    def get_name(self):
        return self.name


class Agent(Item):
    """
    Agent inherits from its parent item, but now it can also perform actions.
    """

    RIGHT = 0
    UP = 1
    LEFT = 2

    RIGHT_FAST = RIGHT + 3
    UP_FAST = UP + 3
    LEFT_FAST = LEFT + 3

    DOWN = 7 # Secret action only for pedestrian
    NULL = -3534
    NB_ACTIONS = 6

    def __init__(self, name, pos, goal, mat, car):
        Item.__init__(self, name, pos)
        self.map = mat

        self.car = car
        self.damage = 0
        self.tiredness = 0
        self.time_taken = 0
        self.succeeds = False
        self.being_civic = False
        self.current_damage = 0
        self.previous_action = -999

        self.move_map = [
            [[Agent.DOWN], [Agent.LEFT], [Agent.LEFT], [Agent.LEFT], [Agent.NULL], [Agent.NULL]],
            [[Agent.DOWN], [Agent.NULL], [Agent.NULL], [Agent.UP], [Agent.NULL], [Agent.NULL]],
            [[Agent.DOWN], [Agent.NULL], [Agent.NULL], [Agent.UP], [Agent.NULL], [Agent.NULL]],
            [[Agent.DOWN], [Agent.LEFT], [Agent.LEFT], [Agent.LEFT], [Agent.NULL], [Agent.NULL]],
            [[Agent.DOWN], [Agent.NULL], [Agent.NULL], [Agent.UP], [Agent.NULL], [Agent.NULL]],
            [[Agent.DOWN], [Agent.NULL], [Agent.NULL], [Agent.UP], [Agent.NULL], [Agent.NULL]],
            [[Agent.DOWN], [Agent.NULL], [Agent.NULL], [Agent.UP], [Agent.NULL], [Agent.NULL]],
            [[Agent.RIGHT], [Agent.RIGHT], [Agent.RIGHT], [Agent.UP], [Agent.UP], [Agent.NULL]],
            [[Agent.NULL], [Agent.NULL], [Agent.NULL], [Agent.NULL], [Agent.NULL], [Agent.NULL]]
        ]

        if constants.degree_of_stochasticity > 0:
             self.move_map[3][3] = [Agent.LEFT, Agent.UP]
        if constants.degree_of_stochasticity > 1:
            self.move_map[5][3] = [Agent.UP, Agent.RIGHT]
        if constants.degree_of_stochasticity > 2:
            self.move_map[3][3] = [Agent.LEFT, Agent.UP, Agent.RIGHT]

    def set_map(self, mat):
        self.map = mat

    def get_map(self):
        return self.map

    def move_request(self, direction_agent, i_insist=None):
        """
        Receives a direction (an integer) and interprets the associated movement.

        :param direction_agent: an integer from 0 to 7
        :return: a move class instance
        """

        origin = self.position[:]
        moved = self

        destination = origin[:]

        velocity = 1

        if direction_agent >= Agent.NB_ACTIONS/2 and self.car:
            velocity = 2

        if direction_agent == Agent.DOWN:
            "Down"
            destination[0] += velocity
        elif direction_agent % (Agent.NB_ACTIONS/2) == Agent.LEFT:
            "Left"
            destination[1] -= velocity
        elif direction_agent % (Agent.NB_ACTIONS/2) == Agent.RIGHT:
            "Right"
            destination[1] += velocity
        elif direction_agent % (Agent.NB_ACTIONS/2) == Agent.UP:
            "Up"
            destination[0] -= velocity
        else:
            print('this ought to never happen')

        if i_insist is not None:
            destination = i_insist

        move = Move(self, moved, origin, destination)

        if move.check_validity(self.map):
            return move
        else:
            return Move(self, moved, origin, origin)

    def act(self):
        """
        A method to randomly choose a possible action.
        :return: a move request.
        """

        direction_chosen = np.random.randint(Agent.NB_ACTIONS)
        return self.move_request(direction_chosen)

    def act_clever(self):
        """
        An heuristic method to choose the best action according to the map and location.
        :return: a move request.
        """

        direction_chosen = self.move_map[self.position[0]][self.position[1]]

        which_one = np.random.randint(len(direction_chosen))
        return self.move_request(direction_chosen[which_one])

    def tire(self):
        self.tiredness += 1

    def get_damaged(self):
        self.damage += 1

    def time_increase(self):
        self.time_taken += 1

    def reset(self):
        self.position = self.origin
        self.succeeds = False
        self.being_civic = False
