package main

import (
//	. "github.com/iHelos/NNBackPropogation/data_helper"
	. "github.com/iHelos/NNBackPropogation/neural_core"
	"fmt"
)

type ImageData struct {
	Label []float64
	Image []float64
}

func main() {
	////layer := NewInputLayer(len(train_x[0]), len(train_y[0]), Sigmoid)
	////layer.Learn(train_x, train_y, 1000, 0.001, 1)
	////
	////test_x := GetTestData("data/test.csv", true)
	//
	//file, _ := os.Create("result.csv")
	//writer := csv.NewWriter(file)
	//for i, value := range test_x {
	//	arr := layer.Predict(value)
	//	label := GetLabel(arr)
	//	writer.Write([]string{strconv.Itoa(i+1), strconv.Itoa(label)})
	//
	//}
	//defer writer.Flush()
	//train_x, _ := GetTrainData("data/train.csv", true)
	nn := CreateNN(
		2,
		LayerMeta{3, SIGMOID},
		LayerMeta{4, TANH},
		LayerMeta{5, SOFTSIGN},
	)
	res := nn.ForwardPropagation([]float64{1,2})
	fmt.Printf("%v", res)
}