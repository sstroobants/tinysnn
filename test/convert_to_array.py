import pandas as pd

filename = "output_weights"

data = pd.read_csv(f"test/{filename}.csv", header=None)

with open(f"test/{filename}_conv.txt", "w") as f:
    f.write("{")
    for i in range(len(data[0])):
        if i != len(data[0]) - 1:
            f.write(f"{data[0][i]:.4f}f, ")
        else:
            f.write(f"{data[0][i]:.4f}f")
    f.write("}")
