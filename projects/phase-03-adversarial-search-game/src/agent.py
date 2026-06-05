class Agent:
    def __init__(self, name="Agent", avatar_path='src/icons/Ankin.png', color=None):
        self.name = name
        self.team = None
        self.avatar_path = avatar_path
        self.avatar = None
        self.color = color

    def set_team(self, team):
        self.team = team

    def reset(self):
        pass

    def get_action(self, state):
        raise NotImplementedError
