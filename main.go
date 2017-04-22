package main

import (
	. "github.com/iHelos/NNBackPropogation/data_helper"
	. "github.com/iHelos/NNBackPropogation/neural_core"
	"fmt"
	_ "net/http/pprof"
	"os"
	"encoding/csv"
	"strconv"
	"log"
	"net/http"
)

type ImageData struct {
	Label []float64
	Image []float64
}

func main() {
	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	train_x, train_y := GetTrainData("data/train.csv", true)
	//fmt.Print(train_y)
	nn := CreateNN(
		len(train_x[0]),
		LayerMeta{200, SIGMOID},
		LayerMeta{len(train_y[0]), SIGMOID},

	)

	test_x := GetTestData("data/test.csv", true)


	nn.Learn(train_x, train_y, 100, 0.5)
	fmt.Printf("%v", nn)


	file, _ := os.Create("result.csv")
	writer := csv.NewWriter(file)
	writer.Write([]string{"ImageId", "Label"})
	for i, value := range test_x {
		arr := GetArrFromProba(nn.ForwardPropagation(value))
		label := GetLabel(arr)
		writer.Write([]string{strconv.Itoa(i+1), strconv.Itoa(label)})

	}
	defer writer.Flush()
}