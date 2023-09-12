import matplotlib.pyplot as plt

# Initialize empty lists to store x and y values
x_values = []
y_values = []

# Read float values from the text file and populate the lists
with open("fitnessValues.txt", "r") as file:
    for line in file:
        # Convert the line (which is a string) back to a float
        float_value = float(line.strip())
        # X-axis will be the index of the value
        x_values.append(len(x_values) + 1)
        y_values.append(float_value)  # Y-axis will be the float value

# Create a line plot
plt.plot(x_values, y_values, marker='o', linestyle='-')

# Add labels and a title
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness vs. Generation")

plt.savefig("plotV11.pdf", format="pdf")

# Display the plot
plt.show()
