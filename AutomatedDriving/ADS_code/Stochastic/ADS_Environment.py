import numpy as np
from ItemAndAgent import Item, Agent
from constants import Values
from window import Window

"""

 Variation of an interesting ethics dilemma with autonomous driven systems.

"""
class Cell:
    def __init__(self, tile):
        self.tile = tile
        self.items = list()

    def get_tile(self):
        return self.tile

    def get_item(self):

        x = -1
        for item in self.items:
            if not isinstance(item, Agent):
                x = item
        return x

    def is_for_garbage(self):
        return self.tile != Environment.IC

    def is_for_car(self):
        return self.tile == Environment.AC or self.tile == Environment.EC

    def is_for_pedestrian(self):
        return self.tile == Environment.EC or self.tile == Environment.PC

    def is_accessible(self, this_is_a_car):
        if this_is_a_car:
            return self.is_for_car()
        else:
            return self.is_for_pedestrian()

    def is_free(self):
        return len(self.items) == 0

    def there_is_an_agent(self):
        for item in self.items:
            if isinstance(item, Agent):
                return True
        return False

    def get_agent(self):
        for item in self.items:
            if isinstance(item, Agent):
                return item
        return 'Sorry'

    def appendate(self, item):
        self.items.append(item)

    def remove(self, item):
        if item not in self.items:
            pass
        else:
            self.items.remove(item)


class Environment:

    AC = 0  # ACCESSIBLE CELL (for cars# )
    IC = 1  # INACCESSIBLE CELL
    PC = 3 # PEDESTRIAN CELL (only for pedestrian)
    EC = 4 # EVERYONE CELL (for everyone)

    agent_goal_1 = [1, 6]
    agent_goal_2 = [2, 6]

    all_actions = range(3 * 2)

    n_objectives = 3

    n_car_states = 0
    n_pedestrian_states = 0

    NB_AGENTS = 1

    individual_objective = 0

    def __init__(self, seed=-1, weights=None,initial_pedestrian_1_cell = 31, initial_pedestrian_2_cell = 45, initial_car_cell = 43):
        self.seed = seed
        self.name = 'Envious'


        self.n_actions = 6
        self.states_agent_left = list()
        self.states_agent_right = list()

        self.weights = weights

        self.map_tileset = create_base_map()

        self.map_length = self.map_tileset.shape[0]
        self.map_width = self.map_tileset.shape[1]
        self.map_num_cells = self.map_length*self.map_width

        self.n_states = self.map_num_cells ** 3

        self.initial_agent_left_position = self.translate_state_cell(initial_car_cell)  # -> 43
        self.initial_pedestrian_1_position = self.translate_state_cell(initial_pedestrian_1_cell)  # -> 31
        self.initial_pedestrian_2_position = self.translate_state_cell(initial_pedestrian_2_cell)

        #print(self.map_width, self.map_length)

        self.map = self.create_cells()

        self.terminal_state_agent_pos1 = self.translate(Environment.agent_goal_1)
        self.terminal_state_agent_pos2 = self.translate(Environment.agent_goal_2)

        #print(self.terminal_state_agent_pos1, self.terminal_state_agent_pos2)

        self.set_states_for_VI()

        self.agents = self.generate_agents(where_left=self.translate_state_cell(initial_car_cell), where_right=self.translate_state_cell(initial_pedestrian_1_cell), where_p2=self.translate_state_cell(initial_pedestrian_2_cell))

        self.items = self.generate_items(first=True)

        self.agent_succeeds = False
        self.internal_damage = False
        self.external_damage = 0

        self.norm_activated = False

        self.window = Window(self.give_window_info())

        self.n_fatalities = 0
        self.n_injuries = 0

        #self.pedestrian_move_map = Agent.move_map


    def set_states_for_VI(self):

        for i in range(len(self.map_tileset)):
            for j in range(len(self.map_tileset[0])):

                if self.map_tileset[i][j] == Environment.PC or self.map_tileset[i][j] == Environment.EC:
                    self.states_agent_right.append(self.translate([i, j]))
                if self.map_tileset[i][j] == Environment.AC or self.map_tileset[i][j] == Environment.EC:
                    if self.translate([i, j]) != self.terminal_state_agent_pos1 and self.translate([i, j]) != self.terminal_state_agent_pos2:
                        self.states_agent_left.append(self.translate([i, j]))
                    else:
                        pass
                        #print("Eliminated because it is terminal :", [i, j])


        #print(self.states_agent_left)
        #print([self.translate_state_cell(i) for i in self.states_agent_left])

        #print("------------------------------------------")

        #print(self.states_agent_right)
        #print([self.translate_state_cell(i) for i in self.states_agent_right])

    def generate_item(self, kind, name, position, goal=None):
        """
        Creates an  item in the game, also modifying the map in order to represent it.
        :param kind: if it is an item or an agent
        :param name: the item/agent's name
        :param position: and its location in the map
        :return:
        """

        if kind == 'Item':
            item = Item(name, position)
        elif kind == 'Pedestrian':
            item = Agent(name, position, goal, self.map_clone(), False)
        else:
            item = Agent(name, position, goal, self.map_clone(), True)

        self.map[position[0], position[1]].appendate(item)

        return item

    def generate_agents(self, where_left=[6, 1], where_right=[4, 3], where_p2=[6, 3], where_goal_left=agent_goal_1, where_goal_right=agent_goal_2):

        agents = list()
        agents.append(self.generate_item('Agent', 8005, where_left, where_goal_left))
        agents.append(self.generate_item('Pedestrian', 2128, where_right, where_goal_right))
        agents.append(self.generate_item('Pedestrian', 2129, where_p2, where_goal_right))

        return agents

    def generate_items(self, mode='hard', first=False):
        """
        Generates all the items/agents in the game.
        :return:
        """
        items = list()

        bumps_positions = [[2, 2], [2, 5], [4, 1]]

        for i in range(len(bumps_positions)):
            if mode == 'hard':
                items.append(self.generate_item('Item', 5, bumps_positions[i]))
            if mode == 'soft':
                if len(self.items) > 0:
                    items.append(self.generate_item('Item', 5, self.items[i].get_position()[:]))
                else:
                    items.append(self.generate_item('Item', 5, bumps_positions[i]))

        return items

    def reset(self, where_agent=-1, where_ped1=-1, where_ped2=-1):
        """
        Returns the game to its original state, before the player or another agent have changed anything.
        :return:
        """

        if where_agent == -1:
            where_agent = self.initial_agent_left_position
        if where_ped1 == -1:
            where_ped1 = self.initial_pedestrian_1_position
        if where_ped2 == -1:
            where_ped2 = self.initial_pedestrian_2_position

        mode = "hard"
        self.map = self.create_cells()
        self.items = self.generate_items(mode)
        self.agents = self.generate_agents(where_agent, where_ped1, where_ped2)

        self.internal_damage = False
        self.external_damage = 0


    def approve_move(self, move):
        """
        Checks if the move goes to an accessible cell.
        To avoid bugs it also checks if the origin of the move makes sense.
        :param move: the move to be checked
        :return: true or false.
        """

        moved_approved = True
        moved_not_approved = False
        origin = move.get_origin()
        destiny = move.get_destination()
        in_between = move.get_in_between()
        fast = move.fast

        ori = self.map[origin[0], origin[1]]
        if fast:
            mid = self.map[in_between[0], in_between[1]]
        dest = self.map[destiny[0], destiny[1]]

        if move.get_moved() == -1:
            # This is what happens when agents stay in the same position. Notice that by doing so
            # they will not be penalised for any value misbehaviour
            return moved_not_approved

        if not ori.is_free():

            if dest.is_for_garbage():
                moved = move.get_moved()

                if dest.is_accessible(moved.car):

                    # Two agents collide or at least they think so
                    if moved.get_map()[destiny[0], destiny[1]].there_is_an_agent() or dest.there_is_an_agent():
                        pass
                    # Agent damaged with garbage
                    elif not dest.is_free() and moved.car:
                        self.internal_damage = True

                    if fast:
                        if moved.get_map()[in_between[0], in_between[1]].there_is_an_agent() or mid.there_is_an_agent():
                            pass
                        # Agent damaged with garbage
                        elif not mid.is_free() and moved.car:
                            self.internal_damage = True

                    return moved_approved
                else:
                    return moved_not_approved

        return moved_not_approved

    def render(self, mode='Training'):
        """
        Until this method is applied the window will not appear.
        :param mode: training or evaluating
        :return:
        """

        self.window.create(mode)

    def give_window_info(self):
        """
        Gives to the window rendering the game the initial information and the information
        that will never change.

        :return:
        """
        return self.map_tileset.copy(), self.agents[0].origin[:], self.agent_goal_1, self.agent_goal_2

    def update_window(self):
        """
        Gives all the needed info to the window rendering the game in order to update it.
        :return:
        """

        info = list()
        info.append(self.agents[0].get_position())

        item_pos = self.agents[1].get_position()[:]
        item_pos.append(self.agents[1].get_name())
        info.append(item_pos)

        item_pos = self.agents[2].get_position()[:]
        item_pos.append(self.agents[2].get_name())
        info.append(item_pos)

        for item in self.items:
            item_pos = item.get_position()[:]
            item_pos.append(item.get_name())
            info.append(item_pos)

        self.window.update(info)

    def drawing_paused(self):
        """
        Wrapper to know if the window rendering has been paused or not.
        :return:
        """
        return self.window.paused

    def do_move_or_not(self, move):
        """
        The method that decides if the move is ultimately approved or not.
        :param move: the move to be checked.
        :return:
        """

        we_approve = self.approve_move(move)

        if we_approve:

            moved = move.get_moved()
            self.remove_from_cell(move.get_origin(), moved)
            self.put_in_cell(move.get_destination(), moved)
            moved.move(move.get_destination())

            mover = move.get_mover()
            mover.tire()

        return we_approve

    def act(self, actions):
        """
        A turn in the environment's game. See step().
        :param actions: the player's action
        :return:
        """
        kill = []
        for agent in self.agents:
            agent.set_map(self.map_clone())

        #shuffled = list(range(len(self.agents)))
        #np.random.shuffle(shuffled)

        move_requests = list()
        external_damage = 0

        for i in range(len(self.agents)):
            while len(actions) < len(self.agents):
                actions.append(8000)

            if actions[i] >= 0:
                if actions[i] == 8000:
                    move_request = self.agents[i].act_clever()
                else:
                    move_request = self.agents[i].move_request(actions[i])

                move_requests.append(move_request)

        if len(move_requests) > 2:
            for i in range(1, len(move_requests)):
                # If both agents want to go to the same place (LETHAL)
                if move_requests[0].get_destination() == move_requests[i].get_destination(): #and self.map[move_requests[0].get_destination()[0], move_requests[0].get_destination()[1]].is_for_car():
                    external_damage += Values.SAFETY_EXTERNAL_LETHAL_MULTIPLIER
                    self.n_fatalities += 1
                    # kill the pedestrian
                    kill.append([True, move_requests[i].get_destination(), self.agents[i]])
                # If the car goes to where the person was (INJURY NOT LETHAL)
                elif move_requests[0].get_destination() == move_requests[i].get_origin():
                    external_damage += Values.SAFETY_EXTERNAL_INJURY_MULTIPLIER
                    self.n_injuries += 1

                if move_requests[0].fast:
                    # If the car traverses where the person goes (LETHAL)
                    if move_requests[0].get_in_between() == move_requests[i].get_destination():
                        external_damage += Values.SAFETY_EXTERNAL_LETHAL_MULTIPLIER
                        self.n_fatalities += 1
                        # kill the pedestrian
                        kill.append([True, move_requests[i].get_destination(), self.agents[i]])
                    # If the car traverses where the person was (INJURY NOT LETHAL)
                    elif move_requests[0].get_in_between() == move_requests[i].get_origin():
                        external_damage += Values.SAFETY_EXTERNAL_INJURY_MULTIPLIER
                        self.n_injuries += 1

        car_aprovat = False
        ##### Order checking finished here
        for i in range(len(move_requests)):
            move_request = move_requests[i]
            move_approved = self.do_move_or_not(move_request)
            if move_approved and i == 0:
                car_aprovat = move_approved
                self.external_damage = external_damage

        if car_aprovat: # i added this line just for the proper use of simulations, can be deleted
            for j in range(len(kill)):
                self.to_ran_over(kill[j][1], kill[j][2])

    def remove_from_cell(self, pos, item):
        self.map[pos[0], pos[1]].remove(item)

    def put_in_cell(self, pos, item):
        self.map[pos[0]][pos[1]].appendate(item)

    def to_ran_over(self, pos, item):
        self.remove_from_cell(pos, item)
        self.put_in_cell([7, 4], item)
        item.move([7, 4])


    def to_reward(self, agent, action):
        """
        Translates the current applied action into its corresponding reward.
        :return: an integer representing the reward.
        """

        speed_multiplier = 1

        if action >= 3:
            speed_multiplier = 1

        reward = [0.0, 0.0, 0.0]
        # Individual objective
        if (agent.get_position() == self.agent_goal_2) or (agent.get_position() == self.agent_goal_1):
            reward[0] += 14.0
            agent.succeeds = True
        else:
            reward[0] += -1.0

        # Value Internal Safety
        if self.internal_damage:
            self.internal_damage = False
            reward[1] += Values.SAFETY_INTERNAL*speed_multiplier  # penalising for getting hurt

        # Value External Safety
        if self.external_damage > 0:
            reward[2] += self.external_damage*Values.SAFETY_EXTERNAL
            self.external_damage = 0

        return np.array(reward)

    def improving(self):
        pos = self.items[0].get_position()[1]

        if pos == 3:
            pos = 1

        if pos == self.where_garbage:
           # print(pos)
            self.in_which_wastebasket[pos] += 1
        else:
            self.original_garbage_position[pos] += 1
            self.in_which_wastebasket[pos] += 1

    def civility_calculator(self):

        results = [0, 0]
        for i in range(len(self.in_which_wastebasket)):

            if self.original_garbage_position[i] == 0:
                results[i] = -1.0
            else:
                results[i] = float(self.in_which_wastebasket[i])/float(self.original_garbage_position[i])
        return results

    def get_state(self):
        """
        Wrapper to get the needed information of the state to the q-learning algorithm.
        :return:
        """
        stator = list()

        for item in self.agents:
            if item.car:
                stator.append(self.translate(item.get_position(), 1))
            else:
                stator.append(self.translate(item.get_position()))

        # We do not care which pedestrian is each
        if len(self.agents) == 3:
            if stator[1] < stator[2]:
                temp = stator[1]
                stator[1] = stator[2]
                stator[2] = temp

        return np.array(stator)

    def step(self, actions):
        """
        Produces a step in the game, letting every agent try to act consecutively.
        :param action: the action that the player will perform
        :return: the information needed for the q-learning algorithm, copying the GYM style.
        """
        self.act(actions.copy())

        rewards = list()
        dones = list()
        agent = self.agents[0]
        action = actions[0]

        reward = self.to_reward(agent, action)

        done = False
        if (agent.get_position() == self.agent_goal_2) or (agent.get_position() == self.agent_goal_1):
            done = True

        rewards.append(reward)
        dones.append(done)
        if self.internal_damage:
            self.internal_damage=False
        
        # rewards is [array([-1.,  0.,  0.])] and we extract [-, -, -]
        rewards = rewards[0]
        # if self.weights is not None:
        #     rewards = np.dot(np.array(rewards), self.weights)
        return self.get_state(), rewards, dones


    def set_stats(self, episode, r_big, mean_score, fourth=0, fifth=0):
        self.window.stats = episode, r_big, mean_score, fourth, fifth

    def eval_stats(self):

        mean_time = 0
        mean_tiredness = 0
        mean_damage = 0

        n = 0
        for agent in self.agents:
            n += 1
            mean_time += agent.time_taken
            mean_tiredness += agent.tiredness
            if self.norm_activated:
                mean_damage += 1
                self.norm_activated = False

        mean_time /= n
        mean_tiredness /= n
        mean_damage /= n

        civility = 0

        pos = self.items[0].get_position()
        if pos == self.waste_basket or pos == self.waste_basket2:
            civility = 1

        return mean_time, mean_tiredness, mean_damage, civility

    def create_cells(self):

        map_struct = list()
        for i in range(len(self.map_tileset)):
            map_struct.append(list())
            for j in range(len(self.map_tileset[0])):
                map_struct[i].append(Cell(self.map_tileset[i, j]))

        return np.array(map_struct)

    def map_clone(self):
        map_struct = list()
        for i in range(len(self.map_tileset)):
            map_struct.append(list())
            for j in range(len(self.map_tileset[0])):
                cell_created = Cell(self.map_tileset[i, j])
                cell_created.items = self.map[i, j].items[:]
                map_struct[i].append(cell_created)

        return np.array(map_struct)

    def translate(self, pos, type=1):
        """
        A method to simplify the state encoding for the q-learning algorithms. Transforms a map 2-dimensional location
        into a 1-dimensional one.
        :param pos: the position in the map (x, y)
        :return: an integer
        """

        return pos[1] + self.map_width * pos[0]

    def translate_state_cell(self, cell, type=0):
        # bastant elegant la verdad.
        #print(self.map_width) -> 7
        return [cell//self.map_width, cell % self.map_width]

    def translate_state(self, celz):

        translated_cells = list()
        for cell in celz:
            translated_cells.append(self.translate_state_cell(cell))

        return translated_cells


def create_base_map():
    """
    The numpy array representing the map.  Change it as you consider it.
    :return:
    """

    base_map = np.array([
        [Environment.PC, Environment.PC, Environment.PC, Environment.PC, Environment.IC, Environment.IC, Environment.IC],
        [Environment.PC, Environment.AC, Environment.AC, Environment.EC, Environment.AC, Environment.AC, Environment.AC],
        [Environment.PC, Environment.AC, Environment.AC, Environment.EC, Environment.AC, Environment.AC, Environment.AC],
        [Environment.PC, Environment.EC, Environment.EC, Environment.PC, Environment.IC, Environment.IC, Environment.IC],
        [Environment.PC, Environment.AC, Environment.AC, Environment.PC, Environment.IC, Environment.IC, Environment.IC],
        [Environment.PC, Environment.AC, Environment.AC, Environment.PC, Environment.IC, Environment.IC, Environment.IC],
        [Environment.PC, Environment.AC, Environment.AC, Environment.PC, Environment.IC, Environment.IC, Environment.IC],
        [Environment.PC, Environment.EC, Environment.EC, Environment.PC, Environment.PC, Environment.IC, Environment.IC],
        [Environment.IC, Environment.IC, Environment.IC, Environment.IC, Environment.IC, Environment.IC, Environment.IC]])

    accessible_cells = (base_map == Environment.AC).sum()
    pedestrian_cells = (base_map == Environment.PC).sum()
    street_cells = (base_map == Environment.EC).sum()

    Environment.n_car_states = accessible_cells + street_cells
    Environment.n_pedestrian_states = pedestrian_cells + street_cells


    return base_map
