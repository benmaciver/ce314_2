import csv

# Open the CSV file for reading
with open('imdb.csv', 'r') as csvfile:
    # Create a csvreader object
    reader = csv.reader(csvfile, delimiter=',')

    # Read the data into a 2D array
    data = []
    for row in reader:
        data.append(row)

def getFirstXEelements(x):
    return data[:x]
    
