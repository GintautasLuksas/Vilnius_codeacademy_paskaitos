
class FitnessTracker:
    def __init__(self, user_name: str, _steps: int):
        self.user_name = user_name
        self._steps = _steps

    def _check_goal(self):
        if self._steps >= 10000:
            print('Daily challange completed!')
    def add_steps(self, addition):
        self._steps += addition
        self._check_goal()

    def reset_steps(self):
        self._steps *= 0
    def get_steps(self):
        return f'Currently you have {self._steps} steps.'



Gintautas = FitnessTracker('Gintautas', 200)
print(Gintautas.get_steps())
Gintautas.reset_steps()
print(Gintautas.get_steps())

Gintautas.add_steps(100000)




