import pandas as pd

df = pd.read_csv("image_assignments.csv")

# Count occurrences per dimension
action_counts = df["Action"].str.split(",").explode().value_counts()
background_counts = df["Background"].str.split(",").explode().value_counts()
object_counts = df["Object"].str.split(",").explode().value_counts()

print("\n--- Action Distribution ---")
print(action_counts)

print("\n--- Background Distribution ---")
print(background_counts)

print("\n--- Object Distribution ---")
print(object_counts)
