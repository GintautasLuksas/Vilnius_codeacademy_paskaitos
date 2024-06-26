class Teacher:
    def __init__(self, name: str, subject: str):
        self.name = name
        self.subject = subject


class Classroom:
    def __init__(self, teachers: list, learners: list):
        self.teachers = teachers
        self.learners = learners

    def add_learner(self, addition):
        self.learners.append(addition)

    def remove_learner(self, remove):
        if remove in self.learners:
            self.learners.remove(remove)

    def display_class(self):
        teacher_info = [(teacher.name, teacher.subject) for teacher in self.teachers]
        return teacher_info, self.learners


teacher1 = Teacher('Nijole', 'Maths')
teacher2 = Teacher('Alma', 'Sports')

class_a = Classroom([teacher1, teacher2], ['Adomas', 'Jore', 'Evelina'])

print(class_a.display_class())
class_a.add_learner('Gintautas')
print(class_a.display_class())
class_a.remove_learner('Jore')
print(class_a.display_class())
