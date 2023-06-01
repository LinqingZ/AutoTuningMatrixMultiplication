from multiprocessing.connection import Client
from Data.storeData import store_data_to_json

def calculate_matrix_in_algorithms():
    pass
rows, columns, matrix, stride = Client.client.rows, Client.client.columns, Client.client.matrix, Client.client.stride
filename = 'matrix_data.json'
store_data_to_json(matrix, filename)