import numpy as np
from map_generator import *
from alg_parameters import *
import matplotlib.pyplot as plt
from util import getFreeCell, returnAsType, renderWorld, Status, get_connected_region
from itertools import combinations
import math

class Agent():
    dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 
               4: (-1, 0), 5: (1, 1), 6: (1, -1), 7: (-1, -1), 8: (-1, 1)}  # x,y operation for corresponding action
    
    oppositeAction = {0:-1, 1:3, 2:4, 3:1, 4:2}

    def __init__(self, world):
        self.__position = np.array([-1,-1])
        self.__goal = np.array([-1,-1])
        self.__emulatedStep = np.array([-1,-1])

        self.bfsMap = None
        self.heuristicMap = None

        self.svo_avg = None
        self.svo_exe = None
        self.svo_ipm = None
        self.svo_distri = 1 / EnvParameters.N_SVOs * np.ones(EnvParameters.N_SVOs)

        self.invalidActions = []
        # static invalid Actions

        self.world = world

        self.restrictedAction = dict()
        # otherAgent restricted Actions (represented as {x:[[a, y], ...]} meaning x action is invalid if agent 'a' takes y action simultaneously)

        self.previousAction = -1

        self.unconditionallyGoodActions = list()

        self.bfsMap = None
        self.heuristicMap = None

        self.fixed_neighbor = -1
        self.weighted_detour = None

    def setGoodActions(self, actions):
        self.unconditionallyGoodActions = actions

    def setInvalidActions(self, actions):
        self.invalidActions = actions

    def updateRestrictedPosition(self, action, newRestriction):
        if action in self.restrictedAction:
            if newRestriction not in self.restrictedAction[action]:
                self.restrictedAction[action].append(newRestriction)
        else:
            self.restrictedAction[action] = [newRestriction]

    def setPos(self, pos):
        self.__position = np.array(pos)
        self.invalidActions = [[],[],[],[]]
        self.restrictedAction = dict()
        self.unconditionallyGoodActions = list()
        self.previousAction = -1


    def getPos(self, type='np'):
        return returnAsType(self.__position, type)

    def setGoal(self, goal):
        self.__goal = np.array(goal)

    def getGoal(self, type='np'):
        return returnAsType(self.__goal, type)
    
    def getEmulatedStep(self, type='np'):
        return returnAsType(self.__emulatedStep, type)

    def emulateStep(self, action):
        step = np.array(self.dirDict[action])
        self.__emulatedStep = np.add(self.getPos(), step)
    
    def takeStep(self, action):
        step = np.array(self.dirDict[action])
        self.setPos(np.add(self.getPos(), step))
        self.previousAction = self.oppositeAction[action]


class MapfGym():

    def __init__(self, restore = False, restorePath = 'savedEnv'):
        if not restore:
            self.obstacleMap = random_generator(SIZE_O=EnvParameters.WORLD_SIZE, PROB_O=EnvParameters.OBSTACLE_PROB).astype(int)
            
            self.agentList = [Agent(self.obstacleMap) for i in range(EnvParameters.N_AGENTS)]
            # assign agents and their goals in the map
            self.populateMap()
            # init the svo stuff
            neighbor, array_wsd = self.get_agent_neighbor_and_overlap()
            for i in range(EnvParameters.N_AGENTS):
                agent = self.agentList[i]
                agent.fixed_neighbor = neighbor[i]
                agent.weighted_detour = array_wsd[i, :]
                agent.svo_avg = np.dot(np.arange(EnvParameters.N_SVOs), agent.svo_distri)
                agent.svo_ipm = np.random.choice(range(EnvParameters.N_SVOs), p=agent.svo_distri.ravel())
                agent.svo_exe = (agent.svo_ipm * 5) / 180 * np.pi

        else:
            loaded = np.load(restorePath+'.npz')
            self.obstacleMap = loaded['map']

            self.agentList = [Agent(self.obstacleMap) for i in range(EnvParameters.N_AGENTS)]
            for idx, agent in enumerate(self.agentList):
                agent.setPos(loaded['agents'][idx])
                agent.setGoal(loaded['goals'][idx])

        self.allGoodActions = self.getUnconditionallyGoodActions(returnIsNeeded=True)

    def replicate(self, world, agentPos, goalPos):
        
        self.obstacleMap = np.copy(world)
        for agentIdx, agent in enumerate(self.agentList):
            agent.setPos(agentPos[agentIdx])
            agent.setGoal(goalPos[agentIdx])
        self.allGoodActions = self.getUnconditionallyGoodActions(returnIsNeeded=True)

    def saveEnv(self, savePath = 'savedEnv'):
        map = self.obstacleMap
        goals = []
        agents = []
        for i in self.agentList:
            agents.append(i.getPos())
            goals.append(i.getGoal())
        agents = np.array(agents)
        goals = np.array(goals)

        np.savez_compressed(savePath, map=map, agents=agents, goals = goals)

    def populateMap(self, edgePoints = None):
        if edgePoints is None:
            tempMap = np.copy(self.obstacleMap)
            
            for i in self.agentList:
                i.setPos(getFreeCell(tempMap))
                tempMap[i.getPos(type='mat')] = 2
            tempMap = np.copy(self.obstacleMap)
            tempMap_withagents = np.copy(self.obstacleMap)
            finding_goal = np.ones(EnvParameters.N_AGENTS)
            for i in range(EnvParameters.N_AGENTS):
                temp_i_agent = self.agentList[i]
                tempMap_withagents[temp_i_agent.getPos()[0]][temp_i_agent.getPos()[1]] = i + 1
            agent_regions = dict()
            for i in range(EnvParameters.N_AGENTS):
                agent_i = self.agentList[i]
                agent_pos = agent_i.getPos()
                while finding_goal[i]:
                    valid_tiles = get_connected_region(tempMap_withagents, agent_regions, agent_pos[0], agent_pos[1])
                    x, y = random.choice(list(valid_tiles))
                    if tempMap[x, y] == 0 and tempMap_withagents[x, y] != -1:
                        agent_i.setGoal((x, y))
                        tempMap[agent_i.getGoal(type='mat')] = 3
                        finding_goal[i] = 0
        else:
            np.random.shuffle(edgePoints)
            agentPos = copy.deepcopy(edgePoints)
            while True:
                np.random.shuffle(edgePoints)
                temp = True
                for i in range(EnvParameters.N_AGENTS):
                    if(np.array_equal(agentPos[i], edgePoints[i])):
                        temp = False
                        break 
                if(temp):
                    break
            for agentIdx, agent in enumerate(self.agentList):
                agent.setPos(agentPos[agentIdx])
                agent.setGoal(edgePoints[(agentIdx)])

    def getAgentsSVOexe(self):
        agentsSVOexe = np.zeros(EnvParameters.N_AGENTS, dtype=np.int64)
        for agentIdx in range(EnvParameters.N_AGENTS):
            agent = self.agentList[agentIdx]
            agentsSVOexe[agentIdx] = agent.svo_ipm
        return agentsSVOexe

    def worldWithAgents(self):
        world = np.copy(self.obstacleMap)
        for i,agent in enumerate(self.agentList):
            if not np.any(agent.getPos()<0):
                world[agent.getPos(type='mat')] = i+1

        return world
    
    def worldWithGoals(self):
        world = np.copy(self.obstacleMap)
        for i,agent in enumerate(self.agentList):
            if agent.getGoal()[0]>=0 and agent.getGoal()[1]>=0:
                world[agent.getGoal(type='mat')] = i+1

        return world
    
    def worldWithAgentsAndGoals(self):
        world = np.copy(self.obstacleMap)
        
        for i,agent in enumerate(self.agentList):
            if not np.any(agent.getPos()<0):
                world[agent.getPos(type='mat')] = i+1
            if not np.any(agent.getGoal()<0):
                world[agent.getGoal(type='mat')] = i+1

        return world

    def makeBfsMap(self, agent:Agent):

        bfsMap = np.copy(self.obstacleMap)

        bfsMap[bfsMap==0] = -2
        size = bfsMap.shape
        curr, end = 0,0
        openedList = list()

        value = -1
        node = (-1,-1)

        openedList.append(agent.getGoal('mat'))

        while end<len(openedList):
            end = len(openedList)
            value+=1

            while curr<end:
                node = openedList[curr]
                # print(node)
                curr+=1
                bfsMap[node] = value
                
                if node[0]>0 and (bfsMap[node[0]-1, node[1]]==-2) and ((node[0]-1, node[1]) not in openedList):
                    openedList.append((node[0]-1, node[1]))
                if (node[0]+1)<size[0] and (bfsMap[node[0]+1, node[1]]==-2) and ((node[0]+1, node[1]) not in openedList):
                    openedList.append((node[0]+1, node[1]))
                if node[1]>0 and (bfsMap[node[0], node[1]-1]==-2) and ((node[0], node[1]-1) not in openedList):
                    openedList.append((node[0], node[1]-1))
                if (node[1]+1)<size[1] and (bfsMap[node[0], node[1]+1]==-2) and ((node[0], node[1]+1) not in openedList):
                    openedList.append((node[0], node[1]+1))
        bfsMap[bfsMap==-1] = 1e6
        agent.bfsMap = bfsMap

    def getHeuristicMap(self, agent:Agent):
        if(agent.bfsMap is None):
            self.makeBfsMap(agent)
            
        heuristicMap = np.zeros((4, self.obstacleMap.shape[0], self.obstacleMap.shape[1])).astype(int)
        for x in range(self.obstacleMap.shape[0]):
            for y in range(self.obstacleMap.shape[1]):
                if self.obstacleMap[x, y] == 0:
                    if x > 0 and agent.bfsMap[x-1, y] < agent.bfsMap[x, y]:
                        assert agent.bfsMap[x-1, y] == agent.bfsMap[x, y] - 1
                        heuristicMap[0, x, y] = 1

                    if x < self.obstacleMap.shape[0] - 1 and agent.bfsMap[x + 1, y] < agent.bfsMap[x, y]:
                        assert agent.bfsMap[x + 1, y] == agent.bfsMap[x, y] - 1
                        heuristicMap[1, x, y] = 1

                    if y > 0 and agent.bfsMap[x, y - 1] < agent.bfsMap[x, y]:
                        assert agent.bfsMap[x, y - 1] == agent.bfsMap[x, y] - 1
                        heuristicMap[2, x, y] = 1

                    if y < self.obstacleMap.shape[1] - 1 and agent.bfsMap[x, y + 1] < agent.bfsMap[x, y]:
                        assert agent.bfsMap[x, y + 1] == agent.bfsMap[x, y] - 1
                        heuristicMap[3, x, y] = 1
        correction = np.ones((4, self.obstacleMap.shape[0], self.obstacleMap.shape[1])).astype(int)
        heuristicMap = heuristicMap * -1 + correction
        agent.heuristicMap = heuristicMap

    def get_positions(self):
        result = []
        for indexOfAgent in range(0, EnvParameters.N_AGENTS):
            agent = self.agentList[indexOfAgent]
            result.append(tuple(agent.getPos()))
        return result

    def get_goals(self):
        result = []
        for indexOfAgent in range(0, EnvParameters.N_AGENTS):
            agent = self.agentList[indexOfAgent]
            result.append(tuple(agent.getGoal()))
        return result

    def get_agent_neighbor_and_overlap(self):

        def fill_rows(list_of_rows, fill_coordinates):
            max_length = max(
                len(row) for row in list_of_rows
            )  # Find the length of the longest row

            # Iterate through each row in the list
            for i, row in enumerate(list_of_rows):
                if len(row) < max_length:  # If the length is smaller than the longest row
                    fill_count = max_length - len(row)
                    fill_values = fill_coordinates[
                        i % len(fill_coordinates)
                        ]  # Select the coordinates based on row index
                    row.extend(
                        [fill_values] * fill_count
                    )  # Add the coordinates to fill the remaining positions

            return list_of_rows

        def get_agent_paths(world, agent_coords, agent_goals):
            """
            Returns A* path for each agent.

            Args:
                world (2D list): The world grid representing the environment.
                agent_coords (list): List of agent coordinates in the format [[x1, y1], [x2, y2], ...].
                agent_goals (list): List of agent goals in the format [[x1, y1], [x2, y2], ...].

            Returns:
                list: A list of paths, where each path is a list of coordinates [[x1, y1], [x2, y2], ...].

            """
            paths = []  # list of paths for each agent
            for i in range(EnvParameters.N_AGENTS):
                start = tuple(agent_coords[i])
                goal = tuple(agent_goals[i])
                if start != goal:
                    path, _ = astar_4(world, start, goal)
                    path.reverse()
                    path.extend([goal])
                else:
                    path = [goal]
                paths.append(path)
            tuple_goal = list(map(tuple, agent_goals))
            filled_path = fill_rows(paths, tuple_goal)
            return filled_path

        def create_direction(paths):
            """
            DIRECTION MAPPING: [1, 2, 3, 4]: [0, 1], [-1, 0], [0, -1], [1, 0]
            """

            def check(prev, curr):
                """
                Prev: Coordinate of the path ex: [3, 4]
                Curr: After coordinate of the path, ex: [4, 4]
                """
                if curr[0] - prev[0] == 0 and curr[1] - prev[1] == 1:
                    return 1
                elif curr[0] - prev[0] == -1 and curr[1] - prev[1] == 0:
                    return 2
                elif curr[0] - prev[0] == 0 and curr[1] - prev[1] == -1:
                    return 3
                elif curr[0] - prev[0] == 1 and curr[1] - prev[1] == 0:
                    return 4
                else:  # no change in direction?
                    return 0

            # create a zero list whose size is same as paths
            directions = [[0 for _ in path] for path in paths]

            for index, path in enumerate(paths):
                length = len(path)
                if not path or length <= 1:
                    continue  # path
                for i in range(1, length):  # [0, 1], [1, 0], [-1, 0], [0, -1]
                    directions[index][i - 1] = check(path[i], path[i - 1])
                directions[index][length - 1] = 0

            return directions

        def overlapping_squares(paths, directions):
            """
            (1) Add all paths to each coordinate (x,y): [(agent, direction), (agent, direction), ... ]
            (2) For each pair of agents (agent1, agent2) add # of collisions,

            """
            # for all agents, calculate its discounted detour caused by other agents
            weighted_sum_detour = np.zeros((EnvParameters.N_AGENTS, EnvParameters.N_AGENTS, 1))

            # iterate through all paths, finding the agents at each square
            mapping = {}
            for index, path in enumerate(paths):
                agent_num = index + 1  # zero indexed
                for j, (x, y) in enumerate(path):
                    if (x, y) not in mapping:
                        mapping[(x, y)] = set()
                    mapping[(x, y)].add((agent_num, directions[index][j]))

            for overlapping_pos, agents in mapping.items():
                if len(agents) <= 1:
                    continue  # no overlapping paths
                for x, y in combinations(list(agents), 2):
                    agent1, dir1 = x
                    agent2, dir2 = y
                    if dir1 == dir2 or agent1 == agent2:
                        continue  # same direction does not add to detour
                    # else, calculate the weighted detour according to the agent's curr pos and overlapping pos
                    dist_to_agent1 = paths[agent1 - 1].index(overlapping_pos)
                    dist_to_agent2 = paths[agent2 - 1].index(overlapping_pos)
                    weighted_sum_detour[agent1 - 1][agent2 - 1] += EnvParameters.OVERLAP_DECAY ** dist_to_agent1 + EnvParameters.OVERLAP_DECAY ** dist_to_agent2
                    weighted_sum_detour[agent2 - 1][agent1 - 1] += EnvParameters.OVERLAP_DECAY ** dist_to_agent1 + EnvParameters.OVERLAP_DECAY ** dist_to_agent2
            return weighted_sum_detour

        paths = get_agent_paths(self.obstacleMap, self.get_positions(), self.get_goals())
        directions = create_direction(paths)
        weighted_sum_detour = overlapping_squares(paths, directions)
        # based on the weighted sum detour, we determine the neighbor we are going to select
        # reshape the weighted_sum_detour as a np.array with size num_agent * num_agents
        array_wsd = np.array(weighted_sum_detour).squeeze()
        neighbor = []
        count_self_neighbor = 0
        for row in array_wsd:
            if np.all(row == 0):  # if the row only contains zeros, no agent will have overlap path with it
                neighbor.append(np.where((array_wsd == row).all(axis=1))[0][count_self_neighbor])
                count_self_neighbor = count_self_neighbor + 1
            else:
                # Find indices of the maximum values, choose one randomly if there are multiple
                max_indices = np.where(row == row.max())[0]
                random_index = np.random.choice(max_indices)
                neighbor.append(random_index)
        assert len(neighbor) == EnvParameters.N_AGENTS
        """
        The neighbor is a np.array like [agent_id_1. agent_id_2. agent_id_3. agent_id_4]
        """
        neighbor = np.array(neighbor)
        neighbor = neighbor + 1

        return neighbor, array_wsd

    def observe(self, indexOfAgent=-1):
        agent = self.agentList[indexOfAgent]

        if agent.heuristicMap is None:
            self.getHeuristicMap(agent)

        #PART 1: FOV Observations

        top_left = (agent.getPos()[0] - EnvParameters.FOV_SIZE // 2, agent.getPos()[1] - EnvParameters.FOV_SIZE // 2)  # (top, left)
        top_left_heuristic = (agent.getPos()[0] - EnvParameters.FOV_Heuristic // 2,
                              agent.getPos()[1] - EnvParameters.FOV_Heuristic // 2)
        bottom_right_heuristic = (
            top_left_heuristic[0] + EnvParameters.FOV_Heuristic, top_left_heuristic[1] + EnvParameters.FOV_Heuristic)

        observations = np.zeros((NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))  #observations per parameters and FOV Size
        observations[5:, :, :] = 1
        # 0: obs map
        # 1: other Agents
        # 2: own goal
        # 3: agents' in Fov goals
        # 4: svo Other Agents
        # 5: heuri_map1
        # 6: heuri_map2
        # 7: heuri_map3
        # 8: heuri_map4

        world = self.worldWithAgents()
        size = world.shape

        visibleAgents = list()

        for i in range(top_left[0], top_left[0] + EnvParameters.FOV_SIZE):  # top and bottom
            for j in range(top_left[1], top_left[1] + EnvParameters.FOV_SIZE):  # left and right
                
                if i >= size[0] or i < 0 or j >= size[1] or j < 0:
                    # out of boundaries (in obstacle map)
                    observations[0,i - top_left[0], j - top_left[1]] = 1
                    continue
                elif world[i,j] == -1:
                    #obstacle (in obstacle map)
                    observations[0,i - top_left[0], j - top_left[1]] = 1

                elif world[i,j] >0 and world[i,j]== indexOfAgent+1:
                    #self Position (in obstacle map)
                    observations[1,i - top_left[0], j - top_left[1]] = 1
                
                elif world[i,j]>0:
                    # other agents in FOV (in agent Map)
                    visibleAgents.append(world[i,j])
                    observations[1,i - top_left[0], j - top_left[1]] = 1 
                    
                    otherAgent = world[i,j]-1
                    # todo: this part should use svo or svo_avg? or we should ignore this channel?
                    if(self.agentList[otherAgent].svo_avg < agent.svo_avg):
                        observations[4,i - top_left[0], j - top_left[1]] = 1

                if top_left_heuristic[0] <= i <= bottom_right_heuristic[0] and top_left_heuristic[1] <= j <= bottom_right_heuristic[1] and 0 <= i < self.obstacleMap.shape[0] and 0 <= j < self.obstacleMap.shape[1]:
                    observations[5, i - top_left[0], j - top_left[1]] = agent.heuristicMap[0, i, j]
                    observations[6, i - top_left[0], j - top_left[1]] = agent.heuristicMap[1, i, j]
                    observations[7, i - top_left[0], j - top_left[1]] = agent.heuristicMap[2, i, j]
                    observations[8, i - top_left[0], j - top_left[1]] = agent.heuristicMap[3, i, j]



        if(top_left[0]<= agent.getGoal()[0]<top_left[0] + EnvParameters.FOV_SIZE and top_left[1]<= agent.getGoal()[1]<top_left[1] + EnvParameters.FOV_SIZE):
            # own goal in FOV (in own goal frame)
            observations[2,agent.getGoal()[0] - top_left[0], agent.getGoal()[1] - top_left[1]] = 1

        for agentIndex in visibleAgents:
            # print(agentIndex)
            x, y = self.agentList[agentIndex-1].getGoal()
            # projection of visible agents' goal in FOV (in others' goal frame)
            min_node = (max(top_left[0], min(top_left[0] + EnvParameters.FOV_SIZE - 1, x)),
                        max(top_left[1], min(top_left[1] + EnvParameters.FOV_SIZE - 1, y)))
            observations[3,min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        #PART2: Goal Vector and Prev Action
        vector = np.zeros(NetParameters.VECTOR_LEN)

        vector[0] = agent.getGoal()[0] - agent.getPos()[0]  # distance on x axes
        vector[1] = agent.getGoal()[1] - agent.getPos()[1]  # distance on y axes
        vector[2] = (vector[0] ** 2 + vector[1] ** 2) ** .5  # total distance
        if vector[2] != 0:  # normalized
            vector[0] = vector[0] / vector[2]
            vector[1] = vector[1] / vector[2]

        # if(agent.previousAction is None ):
        #     prevAction = 0
        # else:
        prevAction = agent.previousAction
        
        vector[3] = prevAction

        # social value orientation
        svo = agent.svo_distri

        return observations, vector, svo

    def getAllObservations(self):
        allObs = np.zeros((1, EnvParameters.N_AGENTS, NetParameters.NUM_CHANNEL , EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE), dtype=np.float32)
        allVectors = np.zeros((1, EnvParameters.N_AGENTS, NetParameters.VECTOR_LEN), dtype=np.float32)
        allSvos = np.zeros((1, EnvParameters.N_AGENTS, EnvParameters.N_SVOs), dtype=np.float32)
        allCommsIndex = np.zeros((1, EnvParameters.N_AGENTS, EnvParameters.N_AGENTS), dtype=np.float32)

        # calculate the updated neighbor for all agents
        neighbor, weighted_detour = self.get_agent_neighbor_and_overlap()
        # updated the self.fixed_neighbor according to the current neighbor and weighted_detour
        # only if we solve the previous neighbor, we can select a new neighbor
        for i in range(EnvParameters.N_AGENTS):
            agent = self.agentList[i]
            if weighted_detour[i][agent.fixed_neighbor - 1] == 0:
                agent.fixed_neighbor = neighbor[i]
            else:
                pass

        for i in range(0, EnvParameters.N_AGENTS):
            observations, vector, svo = self.observe(i)
            
            allObs[:, i] = observations
            allVectors[:, i] = vector
            allSvos[:, i] = svo

        # According to all agent's neighbor, we determine the comms graph
        adj_mat = np.zeros((EnvParameters.N_AGENTS, EnvParameters.N_AGENTS))
        for i in range(EnvParameters.N_AGENTS):
            agent = self.agentList[i]
            adj_mat[i][agent.fixed_neighbor - 1] = 1
            adj_mat[agent.fixed_neighbor - 1][i] = 1
        for i in range(EnvParameters.N_AGENTS):
            allCommsIndex[:, i, :] = adj_mat[i]
        
        return allObs, allVectors, allSvos, allCommsIndex

    def getInvalidActions(self):

            for agent in self.agentList:
                
                staticInvalidAction = list()

                for i in range(0, EnvParameters.N_ACTIONS):
                    agent.emulateStep(i)
                    pos = agent.getEmulatedStep('mat')

                    if not ((0<= pos[0] < self.obstacleMap.shape[0]) and (0<= pos[1] <self.obstacleMap.shape[1])): ## Falling out of map
                        staticInvalidAction.append(i)
                        continue
                    elif(self.obstacleMap[pos] !=0): ## Running into walls
                        staticInvalidAction.append(i)
                        continue
                agent.setInvalidActions(staticInvalidAction)
            
    def getRestrictedActions(self):
        np.zeros((EnvParameters.N_AGENTS, EnvParameters.N_AGENTS, EnvParameters.N_ACTIONS, EnvParameters.N_ACTIONS))
        # Get set of codependent restricted actions

        #Part1: get possible agents that can collide

        agentsAtRisk = list()

        for i in range(EnvParameters.N_AGENTS):
            for j in range(i+1, EnvParameters.N_AGENTS):
                if(np.sum(np.square(self.agentList[i].getPos() - self.agentList[j].getPos()))) <= 4:
                    agentsAtRisk.append([i,j])

        #Part2: get simultaneous actions which cause collision
        for agentOneIndex,agentTwoIndex in agentsAtRisk:

            agentOne = self.agentList[agentOneIndex]
            agentTwo = self.agentList[agentTwoIndex]
            currentDistance = np.sum(np.square(agentOne.getPos() - agentTwo.getPos()))

            for i in range(EnvParameters.N_ACTIONS):
                agentOne.emulateStep(i)

                #Collision is only possible if the agents get closer or atleast stay at the same distance
                if np.sum(np.square(agentOne.getEmulatedStep() - agentTwo.getPos()))<=currentDistance: 
                    
                    #Now check which corressponding action(if any) of agentTwo causes a vertex collision
                    for j in range(EnvParameters.N_ACTIONS):
                        agentTwo.emulateStep(j)

                        if(np.array_equal(agentOne.getEmulatedStep(), agentTwo.getEmulatedStep())):
                            
                            # Add the to the lists
                            agentOne.updateRestrictedPosition(i, [agentTwoIndex, j])
                            agentTwo.updateRestrictedPosition(j, [agentOneIndex, i])

                    # Also account for swapping collision
                    if(np.array_equal(agentOne.getEmulatedStep(), agentTwo.getPos())):
                        agentOne.updateRestrictedPosition(i, [agentTwoIndex, Agent.oppositeAction[i]])
                        agentTwo.updateRestrictedPosition(Agent.oppositeAction[i], [agentOneIndex, i])
        
    def getUnconditionallyGoodActions(self, returnIsNeeded = False):
        # First get bad actions
        self.getInvalidActions()
        self.getRestrictedActions()

        if(returnIsNeeded):
            allGoodActions = list()

        for agent in self.agentList:
            badActions = list()

            badActions += agent.invalidActions

            for i in agent.restrictedAction:
                badActions.append(i)
            

            goodActions = np.setdiff1d(np.arange(EnvParameters.N_ACTIONS),badActions)
            agent.setGoodActions(goodActions)

            if(returnIsNeeded):
                allGoodActions.append(goodActions)
        
        if(returnIsNeeded):
            return allGoodActions

    def getStaticColl(self, actionStatus):
        return np.where(actionStatus==Status.STATIC_COLLISION.value)[0]

    def getActionStatus(self, actions, svo_output):
        # update the svo_distri/svo_avg of all agents
        for i in range(EnvParameters.N_AGENTS):
            agent = self.agentList[i]
            if (agent.getPos()[0] != agent.getGoal()[0]) or (agent.getPos()[1] != agent.getGoal()[1]):
                agent.svo_distri = svo_output[i]
                agent.svo_avg = np.dot(np.arange(EnvParameters.N_SVOs), agent.svo_distri)
                agent.svo_ipm = np.random.choice(range(EnvParameters.N_SVOs), p=svo_output[i].ravel())
                agent.svo_exe = (agent.svo_ipm * 5) / 180 * np.pi
            else:
                selfless = np.zeros(EnvParameters.N_SVOs)
                selfless[-1] = 1
                agent.svo_distri = selfless
                agent.svo_avg = np.dot(np.arange(EnvParameters.N_SVOs), agent.svo_distri)
                agent.svo_ipm = np.random.choice(range(EnvParameters.N_SVOs), p=agent.svo_distri.ravel())
                agent.svo_exe = (agent.svo_ipm * 5) / 180 * np.pi
        # -------------------------------------------
        assert(len(actions)==EnvParameters.N_AGENTS)
        actions = np.copy(actions)
        agentList = list(self.getSvoOrder()[::-1])
        actionStatus = np.full(shape=EnvParameters.N_AGENTS, fill_value=10)
        
        while not len(agentList) == 0:
            indexOfAgent = agentList.pop(0)
            agent = self.agentList[indexOfAgent]
            action = actions[indexOfAgent]

            try:
                assert(action in range(EnvParameters.N_ACTIONS))
            except:
                print(actions, indexOfAgent, action)
                raise Exception("Well, Shit")
            
            if action in agent.invalidActions: ##This caluses a static collision
                actionStatus[indexOfAgent] = min(Status.STATIC_COLLISION.value, actionStatus[indexOfAgent])
                actions[indexOfAgent] = 0
                if 0 in agent.restrictedAction:
                    for fellowAgent,_ in agent.restrictedAction[0]:
                        if(fellowAgent not in agentList):
                            agentList.append(fellowAgent)
            
            elif action in agent.restrictedAction:
                for fellowAgent,agentAction in agent.restrictedAction[action]: ##Check if this is a restricted action and a collision is being caused due to it
                        if(actions[fellowAgent]==agentAction):

                            if(agent.svo_avg >= self.agentList[fellowAgent].svo_avg):
                                actionStatus[indexOfAgent] = min(Status.AGENT_COLLSION.value, actionStatus[indexOfAgent])
                            if(self.agentList[fellowAgent].svo_avg >= agent.svo_avg):
                                actionStatus[fellowAgent] = min(Status.AGENT_COLLSION.value, actionStatus[fellowAgent])

                            actions[indexOfAgent] = 0
                            if 0 in agent.restrictedAction:
                                for fellowAgent,_ in agent.restrictedAction[0]:
                                    if(fellowAgent not in agentList):
                                        agentList.append(fellowAgent)
                
            if (actionStatus[indexOfAgent] == 10): ## This means this is a valid action. It might have been restricted but the other agent might be performing some other action, hence it is valid.
                actionStatus[indexOfAgent] = Status.VALID.value
                
            if(actionStatus[indexOfAgent] == Status.VALID.value):
                agent.emulateStep(action)
                if(np.array_equal(agent.getEmulatedStep(), agent.getGoal())):
                    actionStatus[indexOfAgent] = Status.REACH_GOAL.value
                
                elif(np.array_equal(agent.getPos(), agent.getGoal()) and actions[indexOfAgent]!=0):
                    actionStatus[indexOfAgent] = Status.LEAVE_GOAL.value
                
                elif(actions[indexOfAgent]==agent.previousAction):
                    actionStatus[indexOfAgent] = Status.REPEAT_ACTION.value
        
        assert not np.any((actionStatus)==10) # I'll cry if this is an error

        return actionStatus, actions

    
    def getNonInvalidActions(self):
        nonInvalidAction = []
        for agentIdx, agent in enumerate(self.agentList):
            nonInvalidAction.append(np.setdiff1d(np.arange(5),agent.invalidActions))

        return nonInvalidAction

    # TODO: Maybe we can remove Blocking if we set the priority of an agent who has reached its goal to minimum
    def getBlockingReward(self, indexOfAgent):
        def astar(world, start, goal, robots):
            """A* function for single agent"""
            # print(start, goal)
            for (i, j) in robots:
                world[i, j] = -1
            try:
                path = astar_4(world, start, goal)
            except Exception as e:
                # print(e)
                path = None
            for (i, j) in robots:
                world[i, j] = 0
            return path
        
        other_agents = []
        other_locations = []
        inflation = 10

        agent = self.agentList[indexOfAgent]

        top_left = (agent.getPos()[0] - EnvParameters.FOV_SIZE // 2,
                    agent.getPos()[1] - EnvParameters.FOV_SIZE // 2)
        bottom_right = (top_left[0] + EnvParameters.FOV_SIZE, top_left[1] + EnvParameters.FOV_SIZE)
        for agentIdx, fellowAgent in enumerate(self.agentList):
            if agentIdx == indexOfAgent:
                continue
            x, y = fellowAgent.getPos()
            if x < top_left[0] or x >= bottom_right[0] or y >= bottom_right[1] or y < top_left[1]:
                # exclude agent not in FOV
                continue
            other_agents.append(agentIdx)
            other_locations.append((x, y))
        num_blocking = 0
        world = np.copy(self.obstacleMap)
        for agentIdx in other_agents:
            fellowAgent = self.agentList[agentIdx]
            other_locations.remove(fellowAgent.getPos('mat'))
            # print(agentIdx, other_locations)
            # before removing
            path_before = astar(world, fellowAgent.getPos('mat'), fellowAgent.getGoal('mat'),
                                     robots=other_locations + [agent.getPos('mat')])
            # print(path_before)
            # after removing
            path_after = astar(world, fellowAgent.getPos('mat'), fellowAgent.getGoal('mat'),
                                     robots=other_locations)
            # print(path_after)



            other_locations.append(fellowAgent.getPos('mat'))
            if path_before is None and path_after is None:
                continue
            if path_before is not None and path_after is None:
                continue
            if (path_before is None and path_after is not None) or (len(path_before) > len(path_after) + inflation):
                num_blocking += 1
        return num_blocking * EnvParameters.BLOCKING_COST, num_blocking

    def calculateReward(self, actions, actionStatus):
        svo_post_rewards = np.zeros((1, EnvParameters.N_AGENTS), dtype=np.float32)
        action_post_rewards = np.zeros((1, EnvParameters.N_AGENTS), dtype=np.float32)
        baseRewards = np.zeros((1, EnvParameters.N_AGENTS), dtype=np.float32)
        blockings = np.zeros((1, EnvParameters.N_AGENTS), dtype=np.float32)
        leaveGoals = np.zeros((1, EnvParameters.N_AGENTS), dtype=np.float32)
        numCollide = np.zeros((1, EnvParameters.N_AGENTS), dtype=np.float32)

        for agentIdx, agent in enumerate(self.agentList):

            if(actions[agentIdx]==0):
                if actionStatus[agentIdx]==Status.REACH_GOAL.value:
                    baseRewards[:,agentIdx] = EnvParameters.GOAL_REWARD
                    if EnvParameters.N_AGENTS < 32:  # do not calculate A* for increasing speed
                        blockingReward, num_blocking = self.getBlockingReward(agentIdx)
                        # print(agentIdx, num_blocking)
                        baseRewards[:, agentIdx] += blockingReward
                        if blockingReward < 0:
                            blockings[:, agentIdx] = num_blocking
                elif actionStatus[agentIdx]==Status.REPEAT_ACTION.value:
                    baseRewards[:,agentIdx] = EnvParameters.IDLE_COST
                elif actionStatus[agentIdx]==Status.VALID.value:
                    baseRewards[:, agentIdx] = EnvParameters.IDLE_COST
                else:
                    baseRewards[:, agentIdx] = EnvParameters.COLLISION_COST
                    numCollide[:, agentIdx]+=1
            
            else:
                if actionStatus[agentIdx] == Status.REACH_GOAL.value:
                    baseRewards[:, agentIdx] = EnvParameters.GOAL_REWARD
                
                elif actionStatus[agentIdx] == Status.STATIC_COLLISION.value or \
                            actionStatus[agentIdx] == Status.AGENT_COLLSION.value:
                    baseRewards[:, agentIdx] = EnvParameters.COLLISION_COST
                    numCollide[:, agentIdx]+=1
                elif actionStatus[agentIdx]==Status.REPEAT_ACTION.value:
                    baseRewards[:,agentIdx] = EnvParameters.ACTION_COST
                else:
                    baseRewards[:, agentIdx] = EnvParameters.ACTION_COST
                    if actionStatus[agentIdx] == Status.LEAVE_GOAL.value:
                        leaveGoals[:, agentIdx] += 1

        # re-assign the rewards according to the svo
        for IndexOfAgent in range(EnvParameters.N_AGENTS):
            agent = self.agentList[IndexOfAgent]
            other_agent_rewards = baseRewards[0][agent.fixed_neighbor - 1]
            svo_post_rewards[:, IndexOfAgent] = (baseRewards[0][IndexOfAgent] + other_agent_rewards) / EnvParameters.IMPORTANCE_SVO
            action_post_rewards[:, IndexOfAgent] = math.cos(agent.svo_exe) * baseRewards[0][IndexOfAgent] + math.sin(agent.svo_exe) * other_agent_rewards

        return svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide
    
    def getTrainValid(self, actions, actionStatus):
        trainValid = np.ones((EnvParameters.N_AGENTS, EnvParameters.N_ACTIONS), dtype=np.float32)

        for idx, agent in enumerate(self.agentList):
            for action in agent.invalidActions:
                trainValid[idx, action] = 0
            if(actionStatus[idx]==Status.AGENT_COLLSION.value):
                trainValid[idx, int(actions[idx])] = 0
            
            if(agent.previousAction!=-1):
                trainValid[idx, agent.previousAction] = 0

        return trainValid


    def isConflict(self, agentActionPairs, nextAgentIdx, nextAgentAction):
        if(nextAgentAction in self.agentList[nextAgentIdx].restrictedAction and len(np.array([x for x in set(tuple(x) for x in agentActionPairs) & set(tuple(x) for x in self.agentList[nextAgentIdx].restrictedAction[nextAgentAction])]))!=0):
                return True
        return False    

    def getSvoOrder(self):
        order = []
        for i in self.agentList:
            order.append(i.svo_avg)
        return np.argsort(order)

    def jointStep(self, actions= None, actionStatus=None):
        # if actionStatus is None:
        #     actionStatus, fixedActions = self.getActionStatus(actions)
        #     actions = fixedActions

        goalsReached = np.zeros(EnvParameters.N_AGENTS)

        for agentIdx, agent in enumerate(self.agentList):
            agent.takeStep(actions[agentIdx])

            if(np.array_equal(agent.getPos(), agent.getGoal())):
                goalsReached[agentIdx] = 1

        self.allGoodActions = self.getUnconditionallyGoodActions(returnIsNeeded=False)

        if(np.array_equal(goalsReached, np.ones_like(goalsReached))):
            done = True
        else:
            done = False

        return goalsReached, done

    def _render(self):
        goals = []
        agents = []

        for i in self.agentList:
            agents.append(i.getPos('mat'))
            goals.append(i.getGoal('mat'))
            

        return renderWorld(world=self.obstacleMap, agents=agents,goals=goals,svoOrder=self.getSvoOrder()+1)

        
