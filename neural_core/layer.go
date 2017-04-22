package neural_core

import (
	"math/rand"
//	"fmt"
)

//import "fmt"

type Activation func(x float64) float64

type layerTraining struct {
	last_output_activation []float64
	last_output            []float64
	delta                  []float64
}

type Layer struct {
	input_size  int
	output_size int
	neurons     [][]float64
	layerTraining
	funcs       *NeuralFunction
}

func NewInputLayer(input_size, output_size int, activate NeuralFunction) (*Layer) {
	weights := make([][]float64, output_size)
	output := make([]float64, output_size)
	activation_output := make([]float64, output_size)
	delta := make([]float64, output_size)

	for i, _ := range weights {
		weights[i] = make([]float64, input_size+1)
		for k := range weights[i] {
			weights[i][k] = 0.01 * rand.Float64()
		}
	}
	return &Layer{
		input_size,
		output_size,
		weights,
		layerTraining{
			activation_output,
			output,
			delta,
		},
		&activate,
	}
}

func (layer *Layer) getNeuronPrediction(data []float64, ind int) float64 {
	//fmt.Println(data)
	result := -layer.neurons[ind][layer.input_size]
	for i := 0; i < layer.input_size; i++ {
	//	fmt.Println(i)
		result += data[i] * layer.neurons[ind][i]
	}
	act_result := layer.funcs.activation(result)
	layer.last_output[ind] = result
	layer.last_output_activation[ind] = act_result
	return act_result
}

func (layer *Layer) getLayerPrediction(data []float64) []float64 {
	result := make([]float64, layer.output_size)
	for i := range layer.neurons {
		result[i] = layer.getNeuronPrediction(data, i)
	}
	return result
}

func (layer *Layer) updateWeights(input []float64, eps float64) {
	//fmt.Println(layer.delta)
	//fmt.Printf("eps: %.2f\n", eps)
	for i := 0; i < layer.output_size; i++ {
		temp := eps * layer.delta[i]
		for j := 0; j < layer.input_size; j++ {
			layer.neurons[i][j] += temp * input[j]
		}
		layer.neurons[i][layer.input_size] = temp
	}
	for i := range layer.delta {
		layer.delta[i] = 0
	}
}