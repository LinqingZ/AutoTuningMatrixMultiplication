from Calculation.calculateMatrix import calculate_matrix


def input_matrix_problem():
    # Input the number of rows and columns
    rows = int(input("Enter the number of rows: "))
    columns = int(input("Enter the number of columns: "))

    # Input the matrix elements
    matrix = []
    for i in range(rows):
        row = []
        for j in range(columns):
            element = int(input(f"Enter the element at position ({i+1}, {j+1}): "))
            row.append(element)
        matrix.append(row)

    # Input the stride
    stride = int(input("Enter the stride: "))

    return rows, columns, matrix, stride


rows, columns, matrix, stride = input_matrix_problem()
# print("Matrix:", matrix)
# print("Stride:", stride)


# Calculate matrix
calculate_matrix(rows, columns, matrix, stride)