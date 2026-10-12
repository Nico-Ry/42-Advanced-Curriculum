
"""Train a linear regression model to predict car prices from mileage."""

import csv
import json
import math
from pathlib import Path


# Configuration
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data.csv"
MODEL_FILE = BASE_DIR / "thetas.json"
HISTORY_FILE = BASE_DIR / "training_history.json"

LEARNING_RATE = 0.01
ITERATIONS = 100_000
HISTORY_INTERVAL = 100
PRINT_INTERVAL = 10_000


def load_dataset():
	"""Read the CSV file and return mileage and price lists."""
	mileages = []
	prices = []

	with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)

		for row in reader:
			mileage = float(row["km"])
			price = float(row["price"])

			# Reject invalid numbers before using them in calculations.
			if not math.isfinite(mileage) or mileage < 0:
				raise ValueError("Invalid mileage in dataset.")
			if not math.isfinite(price):
				raise ValueError("Invalid price in dataset.")

			mileages.append(mileage)
			prices.append(price)

	if not mileages:
		raise ValueError("The dataset is empty.")

	return mileages, prices


def normalize_mileages(mileages):
	"""Scale mileages by their maximum to improve gradient stability."""
	max_mileage = max(mileages)

	if max_mileage == 0:
		raise ValueError("At least one mileage must be positive.")

	normalized = [km / max_mileage for km in mileages]

	return normalized, max_mileage


def calculate_gradients(mileages, prices, theta0, theta1):
	"""Calculate the average gradients for both theta parameters."""
	sum_error = 0.0
	sum_error_mileage = 0.0
	m = len(mileages)

	for km, price in zip(mileages, prices):
		prediction = theta0 + theta1 * km
		error = prediction - price

		sum_error += error
		sum_error_mileage += error * km

	gradient0 = sum_error / m
	gradient1 = sum_error_mileage / m

	return gradient0, gradient1


def calculate_mse(mileages, prices, theta0, theta1):
	"""Calculate mean squared error for the current regression line."""
	total_squared_error = 0.0

	for km, price in zip(mileages, prices):
		prediction = theta0 + theta1 * km
		error = prediction - price
		total_squared_error += error ** 2

	return total_squared_error / len(mileages)


def train_model(
	mileages,
	prices,
	learning_rate=LEARNING_RATE,
	iterations=ITERATIONS,
	history_interval=HISTORY_INTERVAL,
	verbose=False
):
	"""Run gradient descent and return the learned parameters and history."""
	normalized_mileages, max_mileage = normalize_mileages(mileages)

	theta0 = 0.0
	theta1 = 0.0

	iterations_history = []
	mse_history = []

	if not math.isfinite(learning_rate) or learning_rate <= 0:
		raise ValueError("Learning rate must be finite and positive.")

	if iterations <= 0 or history_interval <= 0:
		raise ValueError("Iterations and history interval must be positive.")

	for iteration in range(iterations):
		# Both gradients are calculated using the same theta values.
		gradient0, gradient1 = calculate_gradients(
			normalized_mileages, prices, theta0, theta1
		)

		# Update both parameters simultaneously.
		theta0 -= learning_rate * gradient0
		theta1 -= learning_rate * gradient1

		# Record the error after updating the model.
		current_iteration = iteration + 1

		if (
			current_iteration == 1
			or current_iteration % history_interval == 0
			or current_iteration == iterations
		):
			mse = calculate_mse(
				normalized_mileages, prices, theta0, theta1
			)

			iterations_history.append(current_iteration)
			mse_history.append(mse)

			if verbose and (
				current_iteration == 1
				or current_iteration % PRINT_INTERVAL == 0
			):
				print(f"Iteration {current_iteration}: MSE = {mse:.2f}")

	# Convert the normalized slope into euros per kilometer.
	theta1 /= max_mileage

	history = {
		"iterations": iterations_history,
		"mse": mse_history
	}

	return theta0, theta1, history


def save_results(theta0, theta1, history):
	"""Save the trained model and its learning history as JSON."""
	model = {
		"theta0": theta0,
		"theta1": theta1
	}

	with MODEL_FILE.open("w", encoding="utf-8") as file:
		json.dump(model, file, indent=4)
		file.write("\n")

	with HISTORY_FILE.open("w", encoding="utf-8") as file:
		json.dump(history, file, indent=4)
		file.write("\n")


def main():
	"""Coordinate loading, training, and saving the model."""
	mileages, prices = load_dataset()

	theta0, theta1, history = train_model(
		mileages, prices, verbose=True
	)

	save_results(theta0, theta1, history)

	print(f"theta0: {theta0}")
	print(f"theta1: {theta1}")


if __name__ == "__main__":
	main()
