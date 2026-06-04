class Values:

    SAFETY_INTERNAL = -10  # Second objective
    SAFETY_EXTERNAL = -10  # Third objective

    SAFETY_EXTERNAL_INJURY_MULTIPLIER = 0.3
    SAFETY_EXTERNAL_LETHAL_MULTIPLIER = 1.0


degree_of_stochasticity = 1  # integer between 0 and 3 affecting the patterns of pedestrians

    # if degree_of_stochasticity > 0:
    #     self.move_map[3][3] = [Agent.LEFT, Agent.UP]
    # if degree_of_stochasticity > 1:
    #     self.move_map[5][3] = [Agent.UP, Agent.RIGHT]
    # if degree_of_stochasticity > 2:
    #     self.move_map[3][3] = [Agent.LEFT, Agent.UP, Agent.RIGHT]


