class User:

    def __init__(
        self,
        user_id,
        name,
        semantic_path=None,
        steering_path=None
    ):

        self.user_id = user_id
        self.name = name
        self.semantic_path = semantic_path
        self.steering_path = steering_path