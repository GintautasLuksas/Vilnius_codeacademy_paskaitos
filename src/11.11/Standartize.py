import pandas as pd

# Load data
df = pd.read_csv('C:/Users/BossJore/PycharmProjects/Vilnius_codeacademy_paskaitos/src/11.11/Mall_Customers.csv')

# Drop the CustomerID column as it's not needed for clustering
df = df.drop('CustomerID', axis=1)

# Convert Gender to numerical values (0 for Female, 1 for Male)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Save the modified data to a new CSV file
df.to_csv('C:/Users/BossJore/PycharmProjects/Vilnius_codeacademy_paskaitos/src/11.11/modified_Mall_Customers.csv', index=False)

print("Gender conversion complete and saved to 'modified_Mall_Customers.csv'")
