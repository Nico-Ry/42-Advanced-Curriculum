
"""Predict a car's price using previously trained regression parameters."""

import json
import math
from pathlib import Path


# Locate the saved model relative to this script.
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "thetas.json"


def load_model():
	"""Load theta0 and theta1 from the saved JSON model."""
	try:
		with MODEL_FILE.open("r", encoding="utf-8") as file:
			model = json.load(file)

	except FileNotFoundError:
		# The subject requires zero parameters before training.
		print("No trained model found. Using default parameters.")
		return 0.0, 0.0

	except json.JSONDecodeError as error:
		raise ValueError("The model file contains invalid JSON.") from error

	# Check that the JSON contains two valid numeric parameters.
	try:
		theta0 = float(model["theta0"])
		theta1 = float(model["theta1"])

	except (KeyError, TypeError, ValueError) as error:
		raise ValueError("The model must contain numeric theta0 and theta1.") from error

	if not math.isfinite(theta0) or not math.isfinite(theta1):
		raise ValueError("Model parameters must be finite numbers.")

	return theta0, theta1


def get_mileage():
	"""Ask the user for a valid, non-negative mileage."""
	try:
		mileage = float(input("How many km? "))

	except ValueError as error:
		raise ValueError("Please enter a valid number.") from error

	if not math.isfinite(mileage) or mileage < 0:
		raise ValueError("Mileage must be a finite, non-negative number.")

	return mileage


def estimate_price(mileage, theta0, theta1):
	"""Estimate a car's price using the linear regression equation."""
	return theta0 + theta1 * mileage


def main():
	"""Load the model, request mileage, and display the prediction."""
	try:
		theta0, theta1 = load_model()
		mileage = get_mileage()

	except (ValueError, EOFError) as error:
		print(f"Error: {error}")
		return 1

	prediction = estimate_price(mileage, theta0, theta1)

	print(f"Estimated price: {prediction:.2f} €")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
