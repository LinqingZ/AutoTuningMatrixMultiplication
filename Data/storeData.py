import json

def store_data_to_json(matrix, filename): # runtime, matrix data
    # Convert matrix to JSON-serializable format
    json_matrix = [[float(element) for element in row] for row in matrix]

    # Create dictionary object
    data = {
        'matrix': json_matrix
    }

    # Save data to JSON file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


