
"""Evaluate how well the trained linear regression model fits the dataset."""

import math

from train import load_dataset
from predict import MODEL_FILE, load_model, estimate_price


def calculate_metrics(mileages, prices, theta0, theta1):
	"""Calculate MSE, RMSE, MAE, and R² for the supplied dataset."""
	m = len(prices)
	average_price = sum(prices) / m

	sum_squared_error = 0.0
	sum_absolute_error = 0.0
	sum_total_variation = 0.0

	for km, price in zip(mileages, prices):
		# Calculate the difference between the prediction and actual price.
		prediction = estimate_price(km, theta0, theta1)
		error = prediction - price

		# Accumulate the quantities needed for the evaluation metrics.
		sum_squared_error += error ** 2
		sum_absolute_error += abs(error)
		sum_total_variation += (price - average_price) ** 2

	# Calculate the average squared and absolute prediction errors.
	mse = sum_squared_error / m
	rmse = math.sqrt(mse)
	mae = sum_absolute_error / m

	# R² compares our model with always predicting the average price.
	# It is undefined when every actual price is identical.
	if sum_total_variation > 0:
		r_squared = 1 - (sum_squared_error / sum_total_variation)
	else:
		r_squared = None

	return {
		"mse": mse,
		"rmse": rmse,
		"mae": mae,
		"r_squared": r_squared
	}


def display_metrics(metrics, number_of_cars):
	"""Display the model evaluation results in a readable format."""
	print("Linear Regression - Model Evaluation")
	print("------------------------------------")
	print(f"Number of cars: {number_of_cars}")
	print(f"MSE:  {metrics['mse']:.2f}")
	print(f"RMSE: {metrics['rmse']:.2f} €")
	print(f"MAE:  {metrics['mae']:.2f} €")

	if metrics["r_squared"] is None:
		print("R²:   N/A (all actual prices are identical)")
	else:
		print(f"R²:   {metrics['r_squared']:.4f}")


def main():
	"""Load the training data and model, then evaluate its predictions."""
	try:
		# Unlike predict.py, evaluation requires a trained model.
		if not MODEL_FILE.is_file():
			raise FileNotFoundError(
				"No trained model found. Run train.py first."
			)

		mileages, prices = load_dataset()
		theta0, theta1 = load_model()

		metrics = calculate_metrics(mileages, prices, theta0, theta1)
		display_metrics(metrics, len(prices))

	except (OSError, ValueError) as error:
		print(f"Error: {error}")
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
