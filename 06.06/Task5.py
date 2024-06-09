#5.
#5.1 Įdiekite numpy paketą.
#5.2 Sukurkite klasę pavadinimu DataTransformer.
#5.3 Implementuokite metodus matematinėms operacijoms, transformacijoms ir pertvarkymui su NumPy.
#5.4 Transformuokite pavyzdinį duomenų rinkinį naudodami implementuotus metodus.

import numpy as np

class DataTransformer:
    @staticmethod
    def add_scalar(data, scalar):
        return data + scalar

    @staticmethod
    def multiply_scalar(data, scalar):
        return data * scalar

    @staticmethod
    def subtract_scalar(data, scalar):
        return data - scalar

    @staticmethod
    def divide_scalar(data, scalar):
        return data / scalar

    @staticmethod
    def transpose(data):
        return np.transpose(data)

    @staticmethod
    def reshape(data, new_shape):
        return np.reshape(data, new_shape)


data = np.array([[1, 2, 3], [4, 5, 6]])

transformer = DataTransformer()


print("Add scalar:")
print(transformer.add_scalar(data, 10))
print("Multiply scalar:")
print(transformer.multiply_scalar(data, 2))


print("Transpose:")
print(transformer.transpose(data))
print("Reshape:")
print(transformer.reshape(data, (3, 2)))
