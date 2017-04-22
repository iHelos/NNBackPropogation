package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"math"
	//"sync"
	"time"
)

func timeTrack(start time.Time, name string) {
	elapsed := time.Since(start)
	fmt.Printf("%s заняло %s\n", name, elapsed)
}

type Activation func(x float64) float64

type Layer struct {
	input_size    	int
	output_size 	int

	weights    	[][]float64
	bias       	[]float64

	activation Activation
}

func NewInputLayer(input_size, output_size int, activate Activation) (*Layer){
	weights := make([][]float64, output_size)
	bias := make([]float64, output_size)
	for i, _ := range weights{
		weights[i] =  make([]float64, input_size)
	}
	return &Layer{input_size, output_size, weights, bias, activate}
}

func (layer *Layer) getResult(data []float64, ind int) float64 {
	result := -layer.bias[ind]
	for i := 0; i < layer.input_size; i++ {
		result += data[i] * layer.weights[ind][i]
	}
	return layer.activation(result)
}


func (layer *Layer) predict_proba(test_x []float64) []float64 {
	result := make([]float64, layer.output_size)
	for i := range layer.weights {
		result[i] = layer.getResult(test_x, i)
	}
	return result
}

func (layer *Layer) predict(test_x []float64) []float64 {
	probs := layer.predict_proba(test_x)
	max_ind := 0
	max_val := 0.
	for i, val := range probs {
		if val > max_val{
			max_val = val
			max_ind = i
		}
	}
	result := make([]float64, len(probs))
	result[max_ind] = 1
	return result
}

func (layer *Layer) fitNeuron(ind int, train_x []float64, y float64, eps float64) {
	result := layer.getResult(train_x, ind)
	delta_bias := eps * (y - result)
	for i := 0; i < layer.input_size; i++ {
		delta_weight := delta_bias * train_x[i]
		layer.weights[ind][i] = layer.weights[ind][i] + delta_weight
	}
	layer.bias[ind] = layer.bias[ind] - delta_bias
}

func (layer *Layer) fit(train_x []float64, train_y []float64, eps float64, verbose_level int) {
	for i := range layer.weights {
		layer.fitNeuron(i, train_x, train_y[i], eps)
//		fmt.Printf("%#v\n", neuron)
	}
	return
}

func compareArrays(arr1 []float64, arr2 []float64) bool{
	result := true
	for i, val := range arr1{
		if val != arr2[i]{
			result = false
		}
	}
	return result
}

func (layer *Layer) learn(train_data [][]float64, train_result [][]float64, epochs int, eps float64, verbose_level int) {
	defer timeTrack(time.Now(), "Обучение")
	for i := 0; i < epochs; i++ {
		func() {
			if verbose_level > 0 {
				fmt.Printf("Эпоха обучения %d\n", i+1)
				defer func(){
					count_valid := 0
					for i, train_x := range train_data {
						if compareArrays(layer.predict(train_x), train_result[i]){
							count_valid++
						}
					}
					fmt.Printf("Совпало: %d; Точность: %.2f%%\n", count_valid, float64(count_valid)/float64(len(train_data))*100)
				}()
			}
			count := 0
			for i, train_x := range train_data {
				count ++
				layer.fit(train_x, train_result[i], eps, verbose_level)
			}
			fmt.Println(count)
		}()
	}
}

type ImageData struct {
	Label []float64
	Image []float64
}

func Stepper (x float64) float64{
	if x>0{
		return 1.
	} else {
		return 0.
	}
}

func Sigmoid (x float64) float64{
	return 1.0 / (1.0 + math.Exp(-x))
}

func main() {
	train_x, train_y := getTrainData("data/train.csv", true)
	layer := NewInputLayer(len(train_x[0]), len(train_y[0]), Sigmoid)
	layer.learn(train_x, train_y, 1000, 0.001, 1)

	test_x := getTestData("data/test.csv", true)

	file, _ := os.Create("result.csv")
	writer := csv.NewWriter(file)
	for i, value := range test_x {
		arr := layer.predict(value)
		label := getLabel(arr)
		writer.Write([]string{strconv.Itoa(i+1), strconv.Itoa(label)})

	}
	defer writer.Flush()
}

const MAX_DARKNESS float64 = 255

func getTrainData(path string, header bool) (train_x [][]float64, train_y [][]float64) {
	f, _ := os.Open(path)
	r := csv.NewReader(bufio.NewReader(f))
	if header {
		r.Read()
	}
	for {
		record, err := r.Read()
		// Stop at EOF.
		if err == io.EOF {
			break
		}
		value, err := strconv.Atoi(record[0])
		value_arr := make([]float64, 10)
		value_arr[value] = 1.
		if err != nil {
			fmt.Println("Not valid csv file")
			os.Exit(1)
		}
		image := record[1:]
		image_normalized := make([]float64, len(image))
		for ind, pix := range image {
			pix, err := strconv.Atoi(pix)
			if err != nil {
				fmt.Println("Not valid csv file")
				os.Exit(1)
			}
			image_normalized[ind] = float64(pix) / MAX_DARKNESS
		}
		train_y = append(train_y, value_arr)
		train_x = append(train_x, image_normalized)
	}
	return
}

func getTestData(path string, header bool) (test_x [][]float64) {
	f, _ := os.Open(path)
	r := csv.NewReader(bufio.NewReader(f))
	if header {
		r.Read()
	}
	for {
		record, err := r.Read()
		// Stop at EOF.
		if err == io.EOF {
			break
		}
		image := record[0:]
		image_normalized := make([]float64, len(image))
		for ind, pix := range image {
			pix, err := strconv.Atoi(pix)
			if err != nil {
				fmt.Println("Not valid csv file")
				os.Exit(1)
			}
			image_normalized[ind] = float64(pix) / MAX_DARKNESS
		}
		test_x = append(test_x, image_normalized)
	}
	return
}

func getLabel(arr []float64) (label int) {
	for i, value := range arr{
		if value == 1{
			label = i
			return
		}
	}
	return
}