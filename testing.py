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


# front_percep = np.concatenate(
#     original_array[0:-2, 0], [original_array[1, 2]])

front_percep = original_array[0:-1, 1:4]

left_percep = original_array[:, 0:2]

right_percep = original_array[:, -2:]


print(right_percep)


# Print the result
