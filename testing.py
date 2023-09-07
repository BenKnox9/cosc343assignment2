import numpy as np

# Create your 3x5 numpy array (replace this with your actual data)
original_array = np.array([[1, 2, 3, 4, 5],
                           [6, 7, 8, 9, 10],
                           [11, 12, 13, 14, 15]])

# Get the right row (last column)
right_row = original_array[:, 0]

# Get two values to the left of the bottom right corner
bottom_right_corner = original_array[2, 1]

result = original_array[0:-1, 1:-1]
print(result)


# Combine them into a new array
result_array = np.concatenate((right_row, [bottom_right_corner]))

# Print the result
