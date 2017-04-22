package neural_core

import "math"

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func Sigmoid_derivative(input, output float64) float64 {
	return output * (1.0 - output)
}

func Tanh (x float64) float64 {
	return math.Tanh(x)
}

func Tanh_derivative(input, output float64) float64 {
	return 1 - math.Pow(output, 2)
}

func Arctan (x float64) float64 {
	return math.Atan(x)
}

func Arctan_derivative(input, output float64) float64 {
	return 1 / (math.Pow(input, 2) + 1)
}

func Softsign (x float64) float64 {
	return x / (1 + math.Abs(x))
}

func Softsign_derivative(input, output float64) float64 {
	return input / math.Pow((1 + math.Abs(input)),2)
}