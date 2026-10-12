
"""Explore car-price regression and gradient descent in one interactive window."""

import json
import math

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

from predict import MODEL_FILE, estimate_price, load_model
from train import DATA_FILE, HISTORY_FILE, load_dataset, train_model


DASHBOARD_IMAGE = DATA_FILE.parent / "dashboard.png"

# Keep the iteration count fixed when comparing learning rates.
EXPERIMENT_ITERATIONS = 1_000
HISTORY_INTERVAL = 100


def load_training_history():
	"""Read the saved convergence history and validate its structure."""
	with HISTORY_FILE.open("r", encoding="utf-8") as file:
		history = json.load(file)

	iterations = history["iterations"]
	mse_values = history["mse"]

	if not iterations or len(iterations) != len(mse_values):
		raise ValueError("Training history is empty or invalid.")

	return history


def create_dashboard(mileages, prices, theta0, theta1, history):
	"""Create both graphs and connect their interactive controls."""

	# Create one window containing two graphs side by side.
	figure, (price_axes, mse_axes) = plt.subplots(
		1, 2, figsize=(15, 8)
	)

	# Leave space below the graphs for the interactive controls.
	figure.subplots_adjust(
		left=0.06,
		right=0.97,
		top=0.90,
		bottom=0.34,
		wspace=0.27
	)

	# Store the parameters currently used by the interactive graphs.
	# Experiments update these values without modifying the saved model.
	current = {
		"theta0": theta0,
		"theta1": theta1,
		"mileage": None
	}

	# ---------------------------------------------------------
	# LEFT GRAPH: Actual car prices and regression line
	# ---------------------------------------------------------

	cars = price_axes.scatter(
		mileages,
		prices,
		label="Actual cars"
	)

	# A straight regression line only needs two endpoints.
	line_km = [min(mileages), max(mileages)]

	regression_line, = price_axes.plot(
		line_km,
		[estimate_price(km, theta0, theta1) for km in line_km],
		label="Regression line"
	)

	# Each residual connects the real price to its predicted price.
	residuals = price_axes.vlines(
		mileages,
		prices,
		[estimate_price(km, theta0, theta1) for km in mileages],
		colors="gray",
		linestyles="dashed",
		alpha=0.5
	)

	residuals.set_visible(False)

	# A red marker will indicate the user's predicted price.
	prediction_marker, = price_axes.plot(
		[], [],
		"X",
		color="red",
		markersize=9,
		label="Your prediction"
	)

	price_axes.set(
		xlabel="Mileage (km)",
		ylabel="Price (€)",
		title="Car prices and regression"
	)

	price_axes.grid(True, alpha=0.3)
	price_axes.legend()

	# ---------------------------------------------------------
	# RIGHT GRAPH: Gradient descent convergence
	# ---------------------------------------------------------

	# Keep the original training history for comparison.
	mse_axes.plot(
		history["iterations"],
		history["mse"],
		label="Saved training",
		alpha=0.65
	)

	# This line will display the results of new experiments.
	experiment_line, = mse_axes.plot(
		[], [],
		label="Experiment",
		linewidth=2
	)

	mse_axes.set(
		xlabel="Training iteration",
		ylabel="Mean squared error",
		title="Gradient descent convergence"
	)

	# The logarithmic scale makes small improvements visible.
	if all(mse > 0 for mse in history["mse"]):
		mse_axes.set_yscale("log")

	mse_axes.set_xlim(0, EXPERIMENT_ITERATIONS)
	mse_axes.grid(True, alpha=0.3)
	mse_axes.legend()

	# ---------------------------------------------------------
	# CONTROLS: Show and hide graphical elements
	# ---------------------------------------------------------

	checkbox_axes = figure.add_axes([0.07, 0.065, 0.19, 0.19])

	checkboxes = CheckButtons(
		checkbox_axes,
		["Actual cars", "Regression line", "Errors"],
		[True, True, False]
	)

	# Each checkbox is associated with a graphical object.
	artists = {
		"Actual cars": cars,
		"Regression line": regression_line,
		"Errors": residuals
	}

	def toggle_visibility(label):
		"""Show or hide the graphical object selected by the user."""
		artist = artists[label]
		artist.set_visible(not artist.get_visible())

		figure.canvas.draw_idle()

	checkboxes.on_clicked(toggle_visibility)

	# ---------------------------------------------------------
	# CONTROLS: Mileage input and live prediction
	# ---------------------------------------------------------

	input_axes = figure.add_axes([0.34, 0.17, 0.16, 0.055])

	mileage_input = TextBox(
		input_axes,
		"Km: ",
		initial=""
	)

	prediction_text = figure.text(
		0.32, 0.09,
		"Enter mileage and press Enter.",
		fontsize=9
	)

	def refresh_prediction():
		"""Update the prediction using the currently selected model."""
		km = current["mileage"]

		if km is None:
			return

		price = estimate_price(
			km,
			current["theta0"],
			current["theta1"]
		)

		if min(mileages) <= km <= max(mileages):
			# Display a red X on the regression line.
			prediction_marker.set_data([km], [price])

			prediction_text.set_text(
				f"Predicted price: {price:.2f} €"
			)
		else:
			# Still predict, but warn about extrapolation.
			prediction_marker.set_data([], [])

			prediction_text.set_text(
				f"Predicted price: {price:.2f} €\n"
				"Outside training range"
			)

		figure.canvas.draw_idle()

	def update_prediction(text):
		"""Validate the mileage entered by the user."""
		try:
			km = float(text)

			if not math.isfinite(km) or km < 0:
				raise ValueError

		except ValueError:
			current["mileage"] = None
			prediction_marker.set_data([], [])
			prediction_text.set_text("Invalid mileage")
			figure.canvas.draw_idle()
			return

		current["mileage"] = km
		refresh_prediction()

	mileage_input.on_submit(update_prediction)

	# ---------------------------------------------------------
	# CONTROLS: Learning rate and retraining
	# ---------------------------------------------------------

	slider_axes = figure.add_axes([0.65, 0.17, 0.17, 0.035])

	alpha_slider = Slider(
		slider_axes,
		"Alpha",
		valmin=0.0001,
		valmax=0.15,
		valinit=0.01,
		valstep=0.001
	)

	button_axes = figure.add_axes([0.85, 0.14, 0.12, 0.075])

	retrain_button = Button(
		button_axes,
		"Retrain"
	)

	result_text = figure.text(
		0.60, 0.08,
		"Choose alpha and click Retrain.",
		fontsize=9
	)

	def retrain(_event):
		"""Retrain in memory and update both graphs."""
		alpha = alpha_slider.val

		result_text.set_text("Training...")
		figure.canvas.draw_idle()

		try:
			# Each experiment starts from theta0 = 0 and theta1 = 0.
			# The training function returns the new model and history.
			new_theta0, new_theta1, new_history = train_model(
				mileages,
				prices,
				alpha,
				EXPERIMENT_ITERATIONS,
				HISTORY_INTERVAL
			)

			final_mse = new_history["mse"][-1]

			# Reject results if gradient descent became unstable.
			if not all(
				math.isfinite(value)
				for value in (new_theta0, new_theta1, final_mse)
			):
				raise ValueError("Training produced non-finite values.")

		except (ValueError, OverflowError, ZeroDivisionError) as error:
			result_text.set_text(f"Training failed: {error}")
			figure.canvas.draw_idle()
			return

		# -----------------------------------------------------
		# Update the current regression model
		# -----------------------------------------------------

		current["theta0"] = new_theta0
		current["theta1"] = new_theta1

		# Move the regression line using the new theta values.
		regression_line.set_ydata([
			estimate_price(km, new_theta0, new_theta1)
			for km in line_km
		])

		# Recalculate the vertical prediction errors.
		# vlines() returns a LineCollection made of segments.
		residuals.set_segments([
			[
				(km, price),
				(km, estimate_price(km, new_theta0, new_theta1))
			]
			for km, price in zip(mileages, prices)
		])

		# Recalculate any mileage prediction entered previously.
		refresh_prediction()

		# Refresh the price graph's limits.
		price_axes.relim()
		price_axes.autoscale_view()

		# -----------------------------------------------------
		# Update the gradient descent graph
		# -----------------------------------------------------

		experiment_line.set_data(
			new_history["iterations"],
			new_history["mse"]
		)

		experiment_line.set_label(
			f"Experiment (alpha = {alpha:.3f})"
		)

		# A logarithmic scale cannot display zero MSE.
		if any(mse <= 0 for mse in new_history["mse"]):
			mse_axes.set_yscale("linear")
		else:
			mse_axes.set_yscale("log")

		mse_axes.relim()
		mse_axes.autoscale_view()
		mse_axes.set_xlim(0, EXPERIMENT_ITERATIONS)
		mse_axes.legend()

		# Display the resulting parameters and final MSE.
		result_text.set_text(
			f"Alpha: {alpha:.3f} | Final MSE: {final_mse:.2f}\n"
			f"theta0: {new_theta0:.2f} | "
			f"theta1: {new_theta1:.6f}"
		)

		figure.canvas.draw_idle()

		# Save the current dashboard without overwriting model data.
		figure.savefig(DASHBOARD_IMAGE, dpi=150)

	retrain_button.on_clicked(retrain)

	# Keep references to interactive controls while the window is open.
	return figure, (
		checkboxes,
		mileage_input,
		alpha_slider,
		retrain_button
	)


def main():
	"""Load the saved model and open the interactive dashboard."""
	try:
		if not MODEL_FILE.is_file():
			raise FileNotFoundError(
				"Run train.py before plotting."
			)

		mileages, prices = load_dataset()
		theta0, theta1 = load_model()
		history = load_training_history()

		figure, widgets = create_dashboard(
			mileages,
			prices,
			theta0,
			theta1,
			history
		)

		# Saving works even when a graphical display is unavailable.
		figure.savefig(DASHBOARD_IMAGE, dpi=150)

		print(f"Saved: {DASHBOARD_IMAGE}")

		if plt.get_backend().lower() == "agg":
			print("No interactive display available; image saved.")
			plt.close(figure)
		else:
			plt.show()

	except (OSError, ValueError, KeyError, TypeError) as error:
		print(f"Error: {error}")
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
