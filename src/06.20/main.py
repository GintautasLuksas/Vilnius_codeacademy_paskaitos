import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('imdb_movies.csv')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
df = df.where(pd.notnull(df), None)


def save_plot(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def show_plot(plot_number):
    if plot_number == 1:
        plt.figure(figsize=(10, 6))
        df['Year'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title('Number of Movies by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Movies')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif plot_number == 2:
        plt.figure(figsize=(10, 6))
        df['Rating'].plot(kind='hist', bins=20, color='green', alpha=0.7)
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
    elif plot_number == 3:
        top_10_movies = df.nlargest(10, 'Rating')
        plt.figure(figsize=(10, 6))
        plt.barh(top_10_movies['Title'], top_10_movies['Rating'], color='teal')
        plt.gca().invert_yaxis()
        plt.title('Top 10 Movies by Rating')
        plt.xlabel('Rating')
        plt.ylabel('Movie Title')
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()
    elif plot_number == 4:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Year'], df['Rating'], color='blue', alpha=0.5)
        plt.title('Scatter Plot: Year vs Rating')
        plt.xlabel('Year')
        plt.ylabel('Rating')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif plot_number == 5:
        df['Length Category'] = pd.cut(df['Duration (minutes)'], bins=[0, 90, 120, 150, float('inf')], labels=['Short', 'Medium', 'Long', 'Very Long'])
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='Length Category', y='Rating', data=df, palette='viridis', inner='quartile')
        plt.title('Rating Distribution by Duration Category')
        plt.xlabel('Duration Category')
        plt.ylabel('Rating')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
    elif plot_number == 6:
        avg_rating_by_year = df.groupby('Year')['Rating'].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(avg_rating_by_year.index, avg_rating_by_year.values, marker='o', linestyle='-', color='orange')
        plt.title('Average Rating by Year')
        plt.xlabel('Year')
        plt.ylabel('Average Rating')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def button_click(plot_number):
    try:
        show_plot(plot_number)
    except Exception as e:
        messagebox.showerror('Error', str(e))


root = tk.Tk()
root.title('IMDB Movie Analysis')


label = tk.Label(root, text="Select a plot to display:")
label.pack()

button1 = tk.Button(root, text="Plot 1: Number of Movies per Year", command=lambda: button_click(1))
button1.pack()

button2 = tk.Button(root, text="Plot 2: Distribution of Ratings", command=lambda: button_click(2))
button2.pack()

button3 = tk.Button(root, text="Plot 3: Top 10 Movies by Rating", command=lambda: button_click(3))
button3.pack()

button4 = tk.Button(root, text="Plot 4: Year vs Rating", command=lambda: button_click(4))
button4.pack()

button5 = tk.Button(root, text="Plot 5: Rating Distribution by Duration", command=lambda: button_click(5))
button5.pack()

button6 = tk.Button(root, text="Plot 6: Average Rating by Year", command=lambda: button_click(6))
button6.pack()

root.mainloop()
