#python -m venv myenv
#1.1 Įdiekite paketą matplotlib.
#1.2 Sukurkite klasę pavadinimu DataVisualizer.
#1.3 Panaudokite metodą plot_data(x, y).
#1.4 Įdiekite metodą save_plot(failo pavadinimas), kad išsaugotumėte brėžinį faile

import matplotlib.pyplot as plt
class DataVisualizer:
    def plot_data(self, x, y):
        plt.plot(x, y)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Data Visualization')

    def save_plot(self, filename):
        plt.savefig(filename)


visualizer = DataVisualizer()


x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

visualizer.plot_data(x, y)


visualizer.save_plot('plot.png')
