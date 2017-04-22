package main

import (
	. "github.com/iHelos/NNBackPropogation/data_helper"
	. "github.com/iHelos/NNBackPropogation/neural_core"
	"fmt"
//	"os"
//	"encoding/csv"
)

type ImageData struct {
	Label []float64
	Image []float64
}

func main() {
	train_x, train_y := GetTrainData("data/train.csv", true)
	fmt.Print(train_y)
	nn := CreateNN(
		len(train_x[0]),
		//LayerMeta{100, SIGMOID},
		//LayerMeta{20, TANH},
		//LayerMeta{20, SIGMOID},
		LayerMeta{len(train_y[0]), SIGMOID},

	)

	//test_x := GetTestData("data/test.csv", true)
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

	nn.Learn(train_x, train_y, 100, 1)
	fmt.Printf("%v", nn)
}