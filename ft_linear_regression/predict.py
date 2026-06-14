import csv

mileages = []
prices = []

with open("data.csv", "r") as file:
	reader = csv.DictReader(file)

	for row in reader:
		mileages.append(float(row["km"]))
		prices.append(float(row["price"]))

print("Number of rows:", len(mileages))

print("Min mileage:", min(mileages))
print("Max mileage:", max(mileages))

print("Min price:", min(prices))
print("Max price:", max(prices))

average_mileage = sum(mileages) / len(mileages)
average_price = sum(prices) / len(prices)

print("Average mileage:", average_mileage)
print("Average price:", average_price)

theta0 = 0
theta1 = 0

mileage = mileages[0]
real_price = prices[0]

prediction = theta0 + theta1 * mileage
error = prediction - real_price

print("Mileage:", mileage)
print("Real price:", real_price)
print("Prediction:", prediction)
print("Error:", error)
