package neural_core

const (
	SIGMOID  = "sigmoid"
	TANH     = "tanh"
	ATAN     = "arctan"
	SOFTSIGN = "softsign"
)

type NeuralFunction struct {
	activation func(x float64) float64
	derivative func(input, output float64) float64
}

var AvailableFunctions = map[string]NeuralFunction{
	SIGMOID:  NeuralFunction{Sigmoid, Sigmoid_derivative},
	TANH:     NeuralFunction{Tanh, Tanh_derivative},
	ATAN:     NeuralFunction{Arctan, Arctan_derivative},
	SOFTSIGN: NeuralFunction{Softsign, Softsign_derivative},
}
