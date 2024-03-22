# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.crossings = None
        self.previous_target = 2  # 0 = bottom, 1 = top, 2 = mid
        self.top_patrol_point = None
        self.bottom_patrol_point = None
        self.has_reached_middle = False
        self.current_target = 2 # eerste target is altijd mid
        self.walls = None
        self.counter = 0
        self.enemy_mid_row = None
        self.sees_enemy = False

    def winning(self, score):
        if self.red and score > 3:
            return True
        elif not self.red and score < -3:
            return True
        else:
            return False

    def flip_winning(self):
        if self.current_target == 0:  # als je aan het onderste patrol point bent, ga naar het midden
            self.current_target = 2
            self.previous_target = 0
        elif self.current_target == 2:  # als je aan het middelste patrol point bent, ga naar het onderste
            self.current_target = 0
            self.previous_target = 2
        elif self.current_target == 1:  # als je aan het bovenste patrol point bent, ga naar het midden
            self.current_target = 2
            self.previous_target = 1

    def which_side(self, pos):
        if pos[0] < self.width // 2:
            return "red"
        else:
            return "blue"

    def find_crossings(self, walls):
        crossings = []
        for y in range(self.height):
            if not walls[self.width // 2][y]:
                if not walls[(self.width // 2) - 1][y]:
                    crossings.append((self.mid_row, y))
        return crossings

    def compute_bottom_patrol_point(self, bottom_patrol_point, walls):
        x = bottom_patrol_point[0]
        y = bottom_patrol_point[1]
        clear_patrol_points = []
        for height in range(y, y + 6):
            if not walls[x][height]:
                clear_patrol_points.append((x, height))
        return max(clear_patrol_points, key=lambda x: x[1]), min(clear_patrol_points, key=lambda x: x[1])

    def compute_top_patrol_point(self, top_patrol_point, walls):
        x = top_patrol_point[0]
        y = top_patrol_point[1]
        clear_patrol_points = []
        for height in range(y - 5, y + 1):
            if not walls[x][height]:
                clear_patrol_points.append((x, height))
        return min(clear_patrol_points, key=lambda x: x[1]), max(clear_patrol_points, key=lambda x: x[1])

    def flip(self):
        if self.current_target == 1:  # als je aan het bovenste patrol point bent, ga naar het midden
            self.current_target = 2
            self.previous_target = 1
        elif self.current_target == 0:  # als je aan het onderste patrol point bent, ga naar het midden
            self.current_target = 2
            self.previous_target = 0
        elif self.current_target == 2:  # als je aan het middelste patrol point bent, ga naar die waar je niet vandaan komt
            if self.previous_target == 1:
                self.current_target = 0
            elif self.previous_target == 0:
                self.current_target = 1

    def surrounding_walls(self, pos, game_walls):
        walls = []
        no_walls = []
        pos_x = pos[0]
        pos_y = pos[1]
        neighbours = [(pos_x - 1, pos_y), (pos_x + 1, pos_y), (pos_x, pos_y - 1), (pos_x, pos_y + 1)]
        for pos in neighbours:
            if pos[0] < 0 or pos[0] > self.width-1 or pos[1] < 0 or pos[1] > self.height-1:
                neighbours.remove(pos)

        for neighbour in neighbours:
            x = neighbour[0]
            y = neighbour[1]
            if game_walls[x][y]:
                walls.append(neighbour)
            else:
                no_walls.append(neighbour)
        return walls, no_walls

    def calc_safe_spaces(self, game_walls):
        dead_ends = []
        tunnels = []
        safe_spaces = []

        if self.red:
            x_coordinates = range(self.width // 2)
        else:
            x_coordinates = range(self.width // 2, self.width)
        y_coordinates = range(self.height)

        for x in x_coordinates:
            for y in y_coordinates:
                walls, no_walls = self.surrounding_walls((x, y), game_walls)
                is_wall = game_walls[x][y]
                if len(walls) == 3 and not is_wall:
                    dead_ends.append((x, y))
                elif len(no_walls) and not is_wall == 2:
                    tunnels.append((x, y))
                elif not is_wall:
                    safe_spaces.append((x, y))

        def safe_tunnel(tunnel, game_walls):
            partially_safe = []
            walls, no_walls = self.surrounding_walls(tunnel, game_walls)
            for neighbour in no_walls:
                if neighbour in safe_spaces and tunnel in partially_safe:
                    safe_spaces.append(tunnel)
                elif neighbour in safe_spaces:
                    partially_safe.append(tunnel)
                elif neighbour in dead_ends:
                    dead_ends.append(tunnel)
                elif neighbour in tunnel:
                    safe_tunnel(neighbour, game_walls)
                    safe_tunnel(tunnel, game_walls)

        for tunnel in tunnels:
            safe_tunnel(tunnel, game_walls)

        return safe_spaces, dead_ends

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.width = game_state.data.layout.width
        self.height = game_state.data.layout.height
        self.mid = (self.width // 2, self.height // 2)
        # Sla de walls layout op
        self.walls = game_state.data.layout.walls

        self.safe_spaces, self.dead_ends = self.calc_safe_spaces(self.walls)

        # Computes the furthest our agent can go to the middle before crossing into enemy territory
        if self.red:
            self.enemy_mid_row = self.mid[0]
            self.mid = (self.mid[0] - 1, self.mid[1])
        elif not self.red:
            self.enemy_mid_row = self.mid[0] - 1

        self.mid_row = self.mid[0]

        if self.walls[self.mid[0]][self.mid[1]]:
            counter = 1
            new_mids = []
            while len(new_mids) == 0:
                for pos in [(self.mid_row, self.mid[1]+counter), (self.mid_row, self.mid[1]-counter)]:
                    if not self.walls[pos[0]][pos[1]]:
                        new_mids.append(pos)
                if len(new_mids) > 0:
                    self.mid = random.choice(new_mids)
                else:
                    counter += 1

        # Sla de oversteekpunten op
        self.crossings = self.find_crossings(self.walls)

        # Computes the border crossings
        lowest_crossing = min(self.crossings, key=lambda x: x[1])
        highest_crossing = max(self.crossings, key=lambda x: x[1])

        self.bottom_patrol_point, self.bottom_attack_point = self.compute_bottom_patrol_point(lowest_crossing, self.walls)
        self.top_patrol_point, self.top_attack_point = self.compute_top_patrol_point(highest_crossing, self.walls)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def calc_enemy_pos(self, gamestate):
        enemies = [gamestate.get_agent_state(i) for i in self.get_opponents(gamestate)]
        list = []
        for i in enemies:
            if i.get_position() is not None:
                list.append(i.get_position())
        return list

    def get_features(self, game_state, action):
        return_amount = 3
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        old_food_list = self.get_food(game_state).as_list()  # list of the old food
        new_food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(new_food_list)  # self.getScore(successor)
        my_pos = successor.get_agent_state(self.index).get_position()
        old_carrying = game_state.get_agent_state(self.index).num_carrying
        new_carrying = successor.get_agent_state(self.index).num_carrying
        distance_from_crossings_list = [self.get_maze_distance(my_pos, border_crossing) for border_crossing in self.crossings]
        enemy_pos_list = self.calc_enemy_pos(successor)
        dist_to_start = self.get_maze_distance(my_pos, self.start)
        distance_from_ghosts = [self.get_maze_distance(my_pos, enemy_pos) for enemy_pos in enemy_pos_list]
        eaten_food = len(old_food_list) - len(new_food_list)
        is_pacman = successor.get_agent_state(self.index).is_pacman
        my_state = successor.get_agent_state(self.index)
        teammates = self.get_team(successor)
        teammates.remove(self.index)
        teammate_positions_list = [successor.get_agent_state(teammate).get_position() for teammate in teammates]
        teammate_pos = teammate_positions_list[0]
        am_scared = False
        dists = []

        if successor.get_agent_state(self.index).scared_timer > 0:
            am_scared = True

        # Computes the distance to the points of interest
        distance_to_middle = self.get_maze_distance(my_pos, self.mid)
        distance_to_bottom_patrol_point = self.get_maze_distance(my_pos, self.bottom_patrol_point)
        distance_to_top_patrol_point = self.get_maze_distance(my_pos, self.top_patrol_point)

        points_list = [self.bottom_patrol_point, self.top_patrol_point, self.mid]
        distances_list = [distance_to_bottom_patrol_point, distance_to_top_patrol_point, distance_to_middle]

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if
                    a.is_pacman and a.get_position() is not None]

        newScaredTimes = [a.scared_timer for a in enemies]

        # wordt true als de agent voor de eerste keer de middenste kolom bereikt
        if my_pos[0] == self.mid_row and not self.has_reached_middle:
            self.has_reached_middle = True

        # verander van entry point als je geblokkeerd wordt
        if len(enemy_pos_list) != 0 and min([util.manhattanDistance(my_pos, ghost_pos) for ghost_pos in enemy_pos_list]) < 6:
            self.sees_enemy = True
        else:
            self.sees_enemy = False

        # patrouilleren
        if self.sees_enemy and self.current_target == my_pos and not is_pacman:
            self.flip()
            features['distance_to_food'] = util.manhattanDistance(my_pos, points_list[self.current_target])
        elif self.sees_enemy and not is_pacman:
            features['distance_to_food'] = util.manhattanDistance(my_pos, points_list[self.current_target])
        else:
            features['distance_to_food'] = 0

        # Bereken hoe veilig het is in die state
        if len(distance_from_ghosts) == 0 or not is_pacman or max(newScaredTimes) > 4:
            features['ghost_score'] = 0
        else:
            if min(distance_from_ghosts) < 2:
                features['ghost_score'] = -15
            elif min(distance_from_ghosts) < 3:
                features['ghost_score'] = -3
            elif min(distance_from_ghosts) < 4:
                features['ghost_score'] = -2
            elif min(distance_from_ghosts) < 5:
                features['ghost_score'] = -1
            elif min(distance_from_ghosts) < 6:
                features['ghost_score'] = 10
            else:
                features['ghost_score'] = 0

        # als je achtervolgd wordt, vermijdt doodlopende paden
        if len(distance_from_ghosts) != 0 and my_pos in self.dead_ends and is_pacman:
            if min(distance_from_ghosts) < 6:
                features['safety_of_pos'] = -7
                features['cross'] = 15
            elif min(distance_from_ghosts) < 5:
                features['safety_of_pos'] = -10
                features['cross'] = 17
            elif min(distance_from_ghosts) < 4:
                features['safety_of_pos'] = -15
                features['cross'] = 19
            elif min(distance_from_ghosts) < 3:
                features['safety_of_pos'] = -18
                features['cross'] = 21
            elif min(distance_from_ghosts) < 2:
                features['safety_of_pos'] = -100
                features['cross'] = 25

        # Compute distance to the nearest food
        if len(new_food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in new_food_list])
            if old_carrying > return_amount or (len(distance_from_ghosts) > 0 and min(distance_from_ghosts) < 3):
                features['distance_to_food'] = dist_to_start
                features['eaten_food'] = 0
            else:
                features['distance_to_food'] = min_distance
                features['eaten_food'] = eaten_food

        # deposit punten als je aan de grens bent
        if old_carrying > 0 and new_carrying == 0:
            features['cross'] = 100

        if self.winning(game_state.data.score):
            features['winner'] = distances_list[1]
            features['distance_to_food'] = 0
            features['eaten_food'] = 0
        else:
            features['winner'] = 0

        # max 1 defender 1 pacman achtervolgen
        if len(invaders) > 0 and self.winning(game_state.data.score):
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            teammate_dists = [self.get_maze_distance(teammate_pos, a.get_position()) for a in invaders]
            if min(dists) >= min(teammate_dists):
                if max(dists) != min(dists):
                    features['invader_distance'] = max(dists)
            else:
                features['invader_distance'] = min(dists)  # voegt de laagste afstand toe aan de "state score"

        # Computes whether we're on defense (1) or offense (0)
        if self.winning(successor.data.score):
            if not is_pacman:
                features['on_defense'] = 1
            else:
                features['on_defense'] = -1
        else:
            features['on_defense'] = 0

        if len(invaders) > 0 and self.winning(game_state.data.score) and am_scared:
            if min(dists) < 3:
                features['keep_distance'] = 100
            else:
                features['keep_distance'] = 0

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'eaten_food': 1, 'return': -1, 'ghost_score': 7,
                'cross': 3, 'safety_of_pos': 30, 'distance_to_new_entry_point': -1, 'avoid': -1, 'winner': -1, 'invader_distance': -40, 'on_defense': 8, 'keep_distance': 8}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    # Function to find crossings between blue and red territory
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        teammates = self.get_team(successor)
        teammates.remove(self.index)
        teammate_positions_list = [successor.get_agent_state(teammate).get_position() for teammate in teammates]
        teammate_pos = teammate_positions_list[0]
        dists = []
        am_scared = False

        if successor.get_agent_state(self.index).scared_timer > 0:
            am_scared = True

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if
                    a.is_pacman and a.get_position() is not None]  # als het none is kunnen we hen niet zien?
        features['num_invaders'] = len(invaders)

        # max 1 defender 1 pacman achtervolgen
        if len(invaders) > 0 and self.winning(game_state.data.score):
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            teammate_dists = [self.get_maze_distance(teammate_pos, a.get_position()) for a in invaders]
            if min(teammate_dists) < min(dists) != max(dists):
                features['invader_distance'] = max(dists)
            else:
                features['invader_distance'] = min(dists)  # voegt de laagste afstand toe aan de "state score"
        elif len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)  # voegt de laagste afstand toe aan de "state score"

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Computes the distance to the points of interest
        distance_to_middle = self.get_maze_distance(my_pos, self.mid)
        distance_to_bottom_patrol_point = self.get_maze_distance(my_pos, self.bottom_patrol_point)
        distance_to_top_patrol_point = self.get_maze_distance(my_pos, self.top_patrol_point)

        distances_list = [distance_to_bottom_patrol_point, distance_to_top_patrol_point, distance_to_middle]

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman or len(distances_list) == 0: features['on_defense'] = 0

        # First, go to the middle, once you get there, start patrolling between the two patrol points
        if my_pos == self.mid and not self.has_reached_middle:
            self.has_reached_middle = True
            self.current_target = random.randrange(2)  # randomly choose a patrol point to go to

        # als je aan een patrol point bent, flip
            # als we winnen, defend samen met attacker
        if (my_pos == self.top_patrol_point and self.current_target == 1) or (
                my_pos == self.bottom_patrol_point and self.current_target == 0) or (
                my_pos == self.mid and self.has_reached_middle and self.current_target == 2):
            if self.winning(game_state.data.score):
                self.flip_winning()
            else:
                self.flip()

        if not self.has_reached_middle:
            features['distance'] = distance_to_middle
        else:
            features['distance'] = distances_list[self.current_target]

        if len(invaders) > 0 and am_scared:
            if min(dists) < 3:
                features['keep_distance'] = 100
            else:
                features['keep_distance'] = 0

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -40, 'stop': -100, 'reverse': -2,
                'distance': -20, 'keep_distance': 8}
