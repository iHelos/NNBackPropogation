package neural_core

import (
	. "github.com/iHelos/NNBackPropogation/timetracker"
	. "github.com/iHelos/NNBackPropogation/data_helper"
	"github.com/go-errors/errors"
	"fmt"
	"math"
	"time"
)

type LayerMeta struct {
	Size       int
	Activation string
}

type NeuralNetwork struct {
	size   int
	layers [](*Layer)
	eps    float64
}

func CreateNN(input_size int, layer_metas ...LayerMeta) *NeuralNetwork {
	nn := NeuralNetwork{}
	nn.size = len(layer_metas)
	nn.layers = make([](*Layer), nn.size)
	last_output := input_size
	for i, v := range layer_metas {
		funcs, ok := AvailableFunctions[v.Activation]
		if !ok {
			panic(errors.New(fmt.Sprintf("Не найдена функция %s", v.Activation)))
		}
		nn.layers[i] = NewInputLayer(last_output, v.Size, funcs)
		last_output = v.Size
	}
	return &nn
}

func (nn *NeuralNetwork) ForwardPropagation(data []float64) []float64 {
	last_output := data
	for _, layer := range nn.layers {
		last_output = layer.getLayerPrediction(last_output)
	}
	return last_output
}

func (nn *NeuralNetwork) BackwardPropagation(expected []float64) {
	last_layer := nn.layers[len(nn.layers)-1]
	layer_errors := make([]float64, last_layer.output_size)
	for i := range last_layer.neurons {
		layer_errors[i] = expected[i] - last_layer.last_output_activation[i]
		last_layer.delta[i] = layer_errors[i] * last_layer.funcs.derivative(
			last_layer.last_output[i],
			last_layer.last_output_activation[i],
		)
	}
	//fmt.Printf("la_errs: %v", layer_errors)
	for i := len(nn.layers) - 2; i >= 0; i-- {
		layer := nn.layers[i]
		layer_errors = make([]float64, layer.output_size)
		for j := range layer.neurons {
			layer_err := 0.0
			for k := range last_layer.neurons {
				layer_err += last_layer.neurons[k][j] * last_layer.delta[k]
			}
			layer_errors[j] = layer_err
		}
		last_layer = layer
		//fmt.Println(layer_errors)
		for k := range last_layer.neurons {
			last_layer.delta[k] = layer_errors[k] * last_layer.funcs.derivative(
				last_layer.last_output[k],
				last_layer.last_output_activation[k],
			)
		}

	}
}

func (nn *NeuralNetwork) UpdateWeights(row []float64) {
	nn.layers[0].updateWeights(row, nn.eps)
	for i := range nn.layers[1:] {
		nn.layers[i+1].updateWeights(nn.layers[i].last_output_activation, nn.eps)
	}
}

func (nn *NeuralNetwork) Learn(train_x [][]float64, train_y [][]float64, epochs int, eps float64) {
	nn.eps = eps
	if len(train_x) != len(train_y) {
		panic(errors.New(fmt.Sprintf("Количество X данных: %d, кол-во Y данных: %d.", len(train_x), len(train_y))))
	}
	for epoch := 0; epoch < epochs; epoch++ {
		func() {
			defer TimeTrack(time.Now(), "Обучение")
			sum_error := 0.
			count_valid := 0
			for i, v := range train_x {
				nn_output := nn.ForwardPropagation(v)
				//fmt.Println(nn_output)
				if CompareArrays(GetArrFromProba(nn_output), train_y[i]) {
					count_valid++
				}
				for j, y := range train_y[i] {
					sum_error += math.Pow((y - nn_output[j]), 2)
				}
				nn.BackwardPropagation(train_y[i])
				nn.UpdateWeights(v)
			}
			fmt.Printf("Эпоха:%d, Среднеквадратичная ошибка:%.2f, Количество совпавших строк: %d (%.2f%%) \n",
				epoch+1,
				sum_error/float64(len(train_x)),
				count_valid,
				float64(count_valid)/float64(len(train_x))*100,
			)
		}()
	}
}
