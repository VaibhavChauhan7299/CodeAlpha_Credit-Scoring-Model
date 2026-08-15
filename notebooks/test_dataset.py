from ucimlrepo import fetch_ucirepo

# Fetch German Credit dataset
dataset = fetch_ucirepo(id=144)

# Features
X = dataset.data.features

# Target
y = dataset.data.targets

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)